import open3d as o3d
import numpy as np
import cv2
import sys, os, argparse, glob
import multiprocessing as mp
import torch
import collections
import matplotlib.cm as cm
from torch import nn
from copy import deepcopy
from pathlib import Path
import logging
####################################################################

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """
    
    def __init__(self, config):
        super().__init__()

        self.default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
        }

        self.config = {**self.default_config, **config}


        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

        # path = Path(__file__).parent / 'weights/superpoint_v1.pth'
        path = './superpoint_v1.pth'#self.config["path"]
        self.load_state_dict(torch.load(path))

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        print('Loaded SuperPoint model')

    def forward(self, data):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        x = self.relu(self.conv1a(data['image']))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'], as_tuple=False)
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h * 8, w * 8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]

        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
        }
######################################################################
def image2tensor(frame, device):
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)


class SuperPointDetector(object):
    def __init__(self, config={}):
        #self.config = self.default_config
        #self.config = {**self.config, **config}
        self.default_config = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
        "remove_borders": 4,
        "cuda": True
        }
        self.config = {**self.default_config, **config}
        #"path": Path(__file__).parent / "superpoint/superpoint_v1.pth",

        self.device = 'cuda' if torch.cuda.is_available() and self.config["cuda"] else 'cpu'
        self.superpoint = SuperPoint(self.config).to(self.device)

    def __call__(self, image):
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_tensor = image2tensor(image, self.device)
        pred = self.superpoint({'image': image_tensor})

        ret_dict = {
            "image_size": np.array([image.shape[0], image.shape[1]]),
            "torch": pred,
            "keypoints": pred["keypoints"][0].cpu().detach().numpy(),
            "scores": pred["scores"][0].cpu().detach().numpy(),
            "descriptors": pred["descriptors"][0].cpu().detach().numpy().transpose()
        }

        return ret_dict

#############################################################


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.default_config = {
        'descriptor_dim': 256,
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2}

        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        # assert self.config['weights'] in ['indoor', 'outdoor']
        # path = Path(__file__).parent
        # path = path / 'weights/superglue_{}.pth'.format(self.config['weights'])
        path ='./superglue_outdoor.pth' #self.config['path']
        self.load_state_dict(torch.load(path))
        #print('Loaded SuperGlue model (\"{}\" weights)'.format(
        #    self.config['weights']))

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }

        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image_size0'])
        kpts1 = normalize_keypoints(kpts1, data['image_size1'])

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, data['scores0'])
        desc1 = desc1 + self.kenc(kpts1, data['scores1'])

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim'] ** .5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return {
            'matches0': indices0,  # use -1 for invalid match
            'matches1': indices1,  # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }
#################################################



class SuperGlueMatcher(object):
    
    def __init__(self, config={}):
        #self.config = self.default_config
        self.default_config = {
        "descriptor_dim": 256,
        "weights": "outdoor",
        "keypoint_encoder": [32, 64, 128, 256],
        "GNN_layers": ["self", "cross"] * 9,
        "sinkhorn_iterations": 100,
        "match_threshold": 0.2,
        "cuda": True}

        self.config = {**self.default_config, **config}
        

        self.device = 'cuda' if torch.cuda.is_available() and self.config["cuda"] else 'cpu'

        #assert self.config['weights'] in ['indoor', 'outdoor']
        #path = Path(__file__).parent
        #path = path / 'superglue/superglue_{}.pth'.format(self.config['weights'])
        
        self.config["path"] = './superglue_outdoor.pth'

        self.superglue = SuperGlue(self.config).to(self.device)

    def __call__(self, kptdescs):
        # setup data for superglue
        data = {}
        data['image_size0'] = torch.from_numpy(kptdescs["ref"]["image_size"]).float().to(self.device)
        data['image_size1'] = torch.from_numpy(kptdescs["cur"]["image_size"]).float().to(self.device)

        if "torch" in kptdescs["cur"]:
            data['scores0'] = kptdescs["ref"]["torch"]["scores"][0].unsqueeze(0)
            data['keypoints0'] = kptdescs["ref"]["torch"]["keypoints"][0].unsqueeze(0)
            data['descriptors0'] = kptdescs["ref"]["torch"]["descriptors"][0].unsqueeze(0)

            data['scores1'] = kptdescs["cur"]["torch"]["scores"][0].unsqueeze(0)
            data['keypoints1'] = kptdescs["cur"]["torch"]["keypoints"][0].unsqueeze(0)
            data['descriptors1'] = kptdescs["cur"]["torch"]["descriptors"][0].unsqueeze(0)
        else:
            data['scores0'] = torch.from_numpy(kptdescs["ref"]["scores"]).float().to(self.device).unsqueeze(0)
            data['keypoints0'] = torch.from_numpy(kptdescs["ref"]["keypoints"]).float().to(self.device).unsqueeze(0)
            data['descriptors0'] = torch.from_numpy(kptdescs["ref"]["descriptors"]).float().to(self.device).unsqueeze(
                0).transpose(1, 2)

            data['scores1'] = torch.from_numpy(kptdescs["cur"]["scores"]).float().to(self.device).unsqueeze(0)
            data['keypoints1'] = torch.from_numpy(kptdescs["cur"]["keypoints"]).float().to(self.device).unsqueeze(0)
            data['descriptors1'] = torch.from_numpy(kptdescs["cur"]["descriptors"]).float().to(self.device).unsqueeze(
                0).transpose(1, 2)

        # Forward !!
        pred = self.superglue.forward(data)

        # get matching keypoints
        kpts0 = kptdescs["ref"]["keypoints"]
        kpts1 = kptdescs["cur"]["keypoints"]

        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().detach().numpy()

        # Sort them in the order of their confidence.
        match_conf = []
        for i, (m, c) in enumerate(zip(matches, confidence)):
            match_conf.append([i, m, c])
        match_conf = sorted(match_conf, key=lambda x: x[2], reverse=True)

        valid = [[l[0], l[1]] for l in match_conf if l[1] > -1]
        v0 = [l[0] for l in valid]
        v1 = [l[1] for l in valid]
        mkpts0 = kpts0[v0]
        mkpts1 = kpts1[v1]

        ret_dict = {
            "ref_keypoints": mkpts0,
            "cur_keypoints": mkpts1,
            "match_score": confidence[v0]
        }

        return ret_dict

#############################################################

def plot_keypoints(image, kpts, scores=None):
    kpts = np.round(kpts).astype(int)

    if scores is not None:
        # get color
        smin, smax = scores.min(), scores.max()
        assert (0 <= smin <= 1 and 0 <= smax <= 1)

        color = cm.gist_rainbow(scores * 0.4)
        color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
        # text = f"min score: {smin}, max score: {smax}"

        for (x, y), c in zip(kpts, color):
            c = (int(c[0]), int(c[1]), int(c[2]))
            cv2.drawMarker(image, (x, y), tuple(c), cv2.MARKER_CROSS, 6)

    else:
        for x, y in kpts:
            cv2.drawMarker(image, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 6)

    return image
###############################################################

###############################################################
class SimpleVO:
    def __init__(self, args):
        #################################
        self.detector=SuperPointDetector({"cuda": 0})
        self.matcher=SuperGlueMatcher({"cuda": 0, "weights": "outdoor"})
        self.kptdescs = {}


        ################################################
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.K = camera_params['K']
        self.dist = camera_params['dist']
        self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, '*.png'))))

        #self.orb=cv.ORB_create()
        #self.bf=cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        self.vis=o3d.visualization.Visualizer()
        self.vis.create_window(width=900,height=900)

        self.curr_pos=np.zeros((3, 1),dtype=np.float64)
        self.curr_rot=np.eye(3,dtype=np.float64)
    
    ################################################################
    def getLineset(self,center, coo):
        points = [center, coo[0],coo[1],coo[2],coo[3]]
        lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[1,4]]
        colors = [[1,0,0] for i in range(len(lines))]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    def get_pyramid(self,r_M,pos,cam_M):
        pos = pos.reshape(3)
        relativeCor = np.array([[0,0,1],[600,0,1],[600,600,1],[0,600,1]]).T
        realCor = np.dot(r_M, np.dot(np.linalg.inv(cam_M), relativeCor)).T + pos
        
        return pos,realCor
    ##############################################################
    def triangulation(self, R, t, pt1, pt2, K):
        ###################################
        pose1 = np.array([[1, 0, 0, 0], 
                          [0, 1, 0, 0], 
                          [0, 0, 1, 0]])
        pose1 = K.dot(pose1)
        ####################################
        pose2 = np.hstack((R, t))
        pose2 = K.dot(pose2)
        
        tripoints = cv2.triangulatePoints(pose1,pose2,pt1.reshape(2,-1),pt2.reshape(2,-1)).reshape(-1, 4)[:,:3]
        
        return tripoints

    ###############################
    def getScale(self, old_3d, new_3d):


        temp=[]

        for i in range(len(old_3d)):
            shiftold3d=old_3d[(i+1)%len(old_3d)]
            oriold3d=old_3d[i]
            s1=(shiftold3d[0]-oriold3d[0])**2+(shiftold3d[1]-oriold3d[1])**2+(shiftold3d[2]-oriold3d[2])**2
            s1=np.sqrt(s1)


            shiftnew3d=new_3d[(i+1)%len(new_3d)]
            orinew3d=new_3d[i]

            s2=(shiftnew3d[0]-orinew3d[0])**2+(shiftnew3d[1]-orinew3d[1])**2+(shiftnew3d[2]-orinew3d[2])**2
            s2=np.sqrt(s2)
            temp.append(s1/s2)
        #############################
        ratio=np.median(temp)
               
        return ratio
    
    ###########################################################
    

    def process_frames(self):
        prev_img = cv2.imread(self.frame_paths[0])
        self.kptdescs["ref"] = self.detector(prev_img)

        for ind, frame_path in enumerate(self.frame_paths[1:]):
            #######################################################3
            #read image
            curr_img = cv2.imread(frame_path)
            self.kptdescs["cur"] = self.detector(curr_img)
            ####################################################
            #Feature Matching
            #kp1,des1=self.orb.detectAndCompute(prev_img, None)
            #kp2,des2=self.orb.detectAndCompute(curr_img, None)
            ##############################################################3
            #matches=self.bf.match(des2,des1)
            #matches=sorted(matches,key=lambda x:x.distance)

            #pts1 = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            #pts2 = np.float32([kp2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

            #estimate essential matrix Ek,k+1
            #E,mask=cv.findEssentialMat(pts2, pts1, self.K, cv.RANSAC, 0.999, 1, None)
            #_, R_curr,t_curr,mask=cv.recoverPose(E, pts2, pts1, cameraMatrix=self.K, mask=mask)
            #######################################################
            matches = self.matcher(self.kptdescs)

            E, mask = cv2.findEssentialMat(matches['cur_keypoints'], matches['ref_keypoints'],cameraMatrix=self.K,
                                           method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R_curr, t_curr, mask = cv2.recoverPose(E, matches['cur_keypoints'], matches['ref_keypoints'],cameraMatrix=self.K, mask=mask)


            #print(matches)
            #if ind!=0:
                ################################################
                #store index to find common set of consecutive frame
            #    old_kp1_ind=[]

            #    old_kp2_ind=[]
            #    new_kp1_ind=[]

            #    new_kp2_ind=[]
            
            #    com_old_kp1_ind=[]
                #################################

            #    common= set([m.trainIdx for m in matches]).intersection([m.queryIdx for m in matches_prev])
            
            #    for m in matches_prev:
            #        if (m.queryIdx in common):
            #            old_kp1_ind.append(m.trainIdx)
            #            old_kp2_ind.append(m.queryIdx)

            #    for m in matches:
            #        if (m.trainIdx in common):
            #            new_kp1_ind.append(m.trainIdx)
            #            new_kp2_ind.append(m.queryIdx)
                        
            #    for m1 in new_kp1_ind:
            #        m0 = old_kp2_ind.index(m1)
            #        com_old_kp1_ind.append(old_kp1_ind[m0])


            #    pts0=np.float32([old_kp1[idx].pt for idx in com_old_kp1_ind]).reshape(-1, 1, 2)
            #    pts1=np.float32([kp1[idx].pt for idx in new_kp1_ind]).reshape(-1, 1, 2)
            #    pts2=np.float32([kp2[idx].pt for idx in new_kp2_ind]).reshape(-1, 1, 2)

                ######################################
                #find triangulation points 
                #goal: to find relative ratio
            #    old_tri = self.triangulation(R_prev, t_prev, pts0, pts1, self.K)
            #    new_tri = self.triangulation(R_curr, t_curr, pts1, pts2, self.K)
                

            #    ratio = self.getScale(old_tri, new_tri)
                #####################################################

            #else:
            #    ratio=1
            

            #get the relative position to the first frame
            self.curr_pos+=self.curr_rot.dot(t_curr)#*ratio
            self.curr_rot=R_curr.dot(self.curr_rot)
                    
            prev_img=curr_img
            matches_prev=matches
            #old_kp1=kp1
            #old_kp2=kp2
            R_prev=R_curr
            t_prev=t_curr
            self.kptdescs["ref"] = self.kptdescs["cur"]
            
            #########################################################  
            #draw the matched points on the current image  
        #    img = cv.drawKeypoints(curr_img, kp2, None, color=(0,0,255))
            img=plot_keypoints(curr_img,self.kptdescs["cur"]["keypoints"],self.kptdescs["cur"]["scores"])
            cv2.imshow('frame', img)

            
            ############################
            #update the new camera pose in open3D window
            center, corners = self.get_pyramid(self.curr_rot,self.curr_pos, self.K)
            line_set = self.getLineset(center,corners)
            
            self.vis.add_geometry(line_set)
            self.vis.poll_events()

            if cv2.waitKey(30) == 27: break
            
        #################################################
        cv2.destroyWindow('frame')  
        self.vis.run()
        self.vis.destroy_window()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',default="./frames/",help='directory of sequential frames')
    parser.add_argument('--camera_parameters',default='camera_parameters.npy',help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.process_frames()