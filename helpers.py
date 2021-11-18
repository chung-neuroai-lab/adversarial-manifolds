import os

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier, EnsembleClassifier
from art.utils import load_cifar10

from mftma.manifold_analysis_correlation import manifold_analysis_corr
from mftma.alldata_dimension_analysis import alldata_dimension_analysis
# from mftma.manifold_simcap_analysis import manifold_simcap_analysis

def MFTMA_analyze_adversarial_representations(args):
    """
    This function is designed to illustrate the adversarial class and exemplar manifold analysis
    of "Neural Population Geometry Reveals the Role of Stochasticity in Robust Perception" on 
    CIFAR10 trained networks. The function compiles everything in 
    MFTMA_analyze_adversarial_representations.ipynb into a single function to: 
     - load a model 
     - construct class or exemplar manifold data 
     - perturb the images
     - extract representations
     - compute MFTMA and other measures
     - save the results
    """
    # convert args to class, if they are not.
    if isinstance(args, dict):
        args = ArgsDict(args)

    # unpack key variables
    seeds = args.seeds                        # how many seeds to compute MFTMA on
    dataset = args.dataset                    # dataset to use (currently only CIFAR10)
    manifold_type = args.manifold_type        # "class" or "exemplar" manifolds
    P = args.P                                # number of manifolds
    M = args.M                                # number of examples per manifold
    N = args.N                                # max number of features
    eps = args.eps                            # adversarial perturbation strength (Linf)
    eps_step_factor = args.eps_step_factor    # used to compute the step size of the attack
    eps_step = eps / eps_step_factor          # the step size of the attack
    max_iter = args.max_iter                  # how many PGD iterations to use
    random = args.random                      # random perturbations instead of PGD
    model_name = args.model_name              # which model to analyze
    device = args.device                      # device to run models on.
    results_dir = 'results'                   # folder for storing results

    # file name for this analysis constructed from args
    file_name = f'model_{model_name}-manifold_{manifold_type}-eps_{eps}-iter_{max_iter}-random_{random}.csv'

    assert manifold_type in ['class', 'exemplar']
    assert model_name in ['CIFAR_ResNet18', 'CIFAR_VOneResNet18']
    
    # load the model
    model = load_model(model_name, loc=device)

    # wrap the model with adversarial robustness toolbox, for generating adversarial stimuli.
    classifier = art_wrap_model(model)
    
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()

    x_train = x_train.transpose(0,3,1,2).astype(np.float32)
    x_test = x_test.transpose(0,3,1,2).astype(np.float32)
    
    # get model clean accuracy
    predictions = classifier.predict(x_test)
    clean_accuracy = accuracy(predictions, y_test)
    print("Accuracy on benign test examples: {}%".format(clean_accuracy * 100))
    
    # construct manifold stimuli
    X, Y = construct_manifold_stimuli(x_test, y_test, manifold_type, P=P, M=M)

    # perturb stimuli
    X_adv = perturb_stimuli(
        X, 
        Y, 
        classifier, 
        eps=eps, 
        eps_step_factor=eps_step_factor, 
        max_iter=max_iter, 
        random=random
    )

    print(f'stimuli shape: {X_adv.shape}')

    # get adversarial accuracy
    adv_accuracy = accuracy(classifier.predict(X_adv), Y)
    print(f"Accuracy on adversarial test examples: {adv_accuracy * 100}")
    
    # apply hooks to the model, to extract intermediate representations
    hooks = {}

    for layer_name, module in model_layer_map(model_name, model).items():
        hooks[layer_name] = Hook(module, layer_name)
        
    # run the perturbed stimuli through the model
    Y_hat = model(torch.tensor(X_adv))

    # put activations and pixels into a dictionary with layer names
    features_dict = {'0.pixels' : X_adv}
    features_dict.update({layer_name: hook.activations for layer_name, hook in hooks.items()})
    
    # run MFTMA analysis on the features
    df = pd.concat([
        MFTMA_analyze_activations(features_dict, P, M, N=N, seed=seed)
        for seed in range(seeds)
    ])
                   
    ## add additional meta data
    df['model'] = model_name
    df['manifold_type'] = manifold_type
    df['eps'] = eps
    df['eps_step'] = eps_step
    df['max_iter'] = max_iter
    df['clean_accuracy'] = clean_accuracy
    df['adv_accuracy'] = adv_accuracy
    df['random'] = random
    
    # store the results
    df.to_csv(os.path.join(results_dir, file_name)) 

class ArgsDict(object):
    """
    convert a dictionary to a class with attrs reflecting keys
    """
    def __init__(self, args_dict):
        for key in args_dict:
            setattr(self, key, args_dict[key])

def load_model(model_name, loc):
    model_locations = {
        'CIFAR_ResNet18': 'models/model_Resnet18-nm_None-dataset_cifar10-epoch_150.ckpt',
        'CIFAR_VOneResNet18': 'models/model_VOneResnet18-nm_gn-nl_0.07-dataset_cifar10-epoch_150.ckpt',

        # coming soon
        #'CIFAR_ATResNet18': 'models/model_Resnet18-AT_linf-eps_8-dataset_cifar10-epoch_150.ckpt'
    }

    return torch.load(model_locations[model_name], map_location=loc)

def art_wrap_model(
    model,
    device_type='gpu',
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
    input_shape=(3, 32, 32),
    clip_values=(0,1),
    nb_classes=10):
    """
    return an art wrapped pytorch model, for generating adversarial stimuli.
    """

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    mean = np.array(mean).reshape((3, 1, 1))
    std = np.array(std).reshape((3, 1, 1))

    return PyTorchClassifier(
        model=model,
        clip_values=clip_values,
        preprocessing=(mean, std),
        loss=criterion,
        optimizer=optimizer,
        input_shape=input_shape,
        nb_classes=nb_classes,
        device_type=device_type,
    )

def accuracy(Y, Y_hat):
    return np.sum(np.argmax(Y, axis=1) == np.argmax(Y_hat, axis=1)) / len(Y)

def construct_manifold_stimuli(X, Y, manifold_type, **kwargs):
    if manifold_type == 'class':
        return construct_class_manifold_stimuli(X, Y, **kwargs)
    elif manifold_type == 'exemplar':
        return construct_exemplar_manifold_stimuli(X, Y, **kwargs)

def construct_class_manifold_stimuli(X, Y, P=10, M=50, flat=True):
    """
    X: test set images
    Y: one hot test set labels
    P: number of classes
    M: number of examples per manifold
    flat: return all stimuli and labels in a single column; else, construct as P by M by stimuli
    returns stimuli and labels
    """
    stimuli_shape = X.shape[1:]
    label_shape = Y.shape[1:]
    
    # sort dataset by classes
    label_sorted_idx = Y.argmax(1).argsort()

    X = X[label_sorted_idx]
    Y = Y[label_sorted_idx]
    
    # construct manifolds P by M by stim / label shape
    X_ = np.zeros((P,M,*stimuli_shape))
    Y_ = np.zeros((P,M,*label_shape))
    
    for p in range(P):
        X_[p] = X[Y.argmax(1)==p][:M]
        Y_[p] = Y[Y.argmax(1)==p][:M]
        
    if flat:
        X_ = X_.reshape(-1, *stimuli_shape)
        Y_ = Y_.reshape(-1, *label_shape)
    
    return X_, Y_

def construct_exemplar_manifold_stimuli(X, Y, P=100, M=50, flat=True):
    """
    X: test set images
    Y: one hot test set labels
    P: number of exemplar manifolds
    M: number of examples per manifold
    flat: return all stimuli and labels in a single column; else, construct as P by M by stimuli
    returns stimuli and labels
    """
    stimuli_shape = X.shape[1:]
    label_shape = Y.shape[1:]
    
    # 100 unique indices
    idxs = np.random.choice(np.arange(X.shape[0]), size=P, replace=False)
    
    X = X[idxs]
    Y = Y[idxs]
    
    # repeat each unique image M times
    X = np.repeat(X, M, axis=0).reshape(P, M, *stimuli_shape)
    Y = np.repeat(Y, M, axis=0).reshape(P, M, *label_shape)
    
    if flat:
        X = X.reshape(-1, *stimuli_shape)
        Y = Y.reshape(-1, *label_shape)
    
    return X, Y

def perturb_stimuli(X, Y, classifier, eps, eps_step_factor=1, max_iter=1, random=False):
    """
    L_inf constrained PGD / FGSM on stimuli
    
    X: stimuli
    Y: labels
    eps: strength of perturbation (l inf)
    eps_step_factor: define the perturbation step_size = eps / eps_step_factor
    max_iter: how many iterations for the attack. 1 for FGSM
    """
    X = X.astype("float32")
    
    if random:
        X_adv = X + (eps*np.sign(np.random.randn(*X.shape))).astype('float32')
    
    else:
        eps_step = eps/eps_step_factor
    
        attack = ProjectedGradientDescent(
            estimator=classifier,
            max_iter=max_iter, 
            norm=np.inf,
            eps=eps,
            eps_step=eps_step,
            targeted=False,
            verbose=False,
            num_random_init=1 # random starting location in eps ball
        )
    
        X_adv = attack.generate(x=X, y=Y)
    
    return X_adv


# A simple hook class to get the output of a layer during forward/backward pass
class Hook():
    def __init__(self, module, name, backward=False):
        self.name = name
        self.activations = None

        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.activations = output.clone().cpu().detach().numpy()

    def close(self):
        self.hook.remove()

# layers to collect features from, and names
def model_layer_map(name, model):
    if name == 'CIFAR_ResNet18':
        return {
            '1.conv1'  : model.conv1,
            '2.block1' : model.layer1,
            '3.block2' : model.layer2,
            '4.block3' : model.layer3,
            '5.block4' : model.layer4,
            '6.linear' : model.linear
        }
    elif name == 'CIFAR_VOneResNet18':
        return {
            "1.VOneBlock.Noise"  : model.conv1.noise,
            '2.block1'           : model.layer1,
            '3.block2'           : model.layer2,
            '4.block3'           : model.layer3,
            '5.block4'           : model.layer4,
            '6.linear'           : model.linear
        }

def MFTMA_analyze_activations(features_dict, P, M, N, kappa=0, NT=100, SIMCAP=False, seed=0, verbose=False):
    """
    Takes in a dictionary of {'layer': features} and processes each with MFTMA. returns a dataframe with the results.
    features_dict: {'layer': features}
    P: number of manifolds
    M: number of examples per manifold
    N: maximum feature dimension (if features > N, random projection will be used)
    kappa: SVM margin
    NT: number of t's for MFTMA
    SIMCAP: also simulate the capacity to measure empirically
    seed: random seed to use
    """
    np.random.seed(seed)
    dfs = []
    for layer, features in features_dict.items():
        X = process_features(features, P, M, N, seed=seed, verbose=verbose)

        # collect MFTMA measures
        capacity_all, radius_all, dimension_all, center_correlation, K = manifold_analysis_corr(X, kappa, NT)
        D_participation_ratio, D_explained_variance, D_feature = alldata_dimension_analysis(X, perc=.9)

        df = pd.DataFrame(
                columns = ['cap', 'dim', 'rad'],
                data = np.array([
                        capacity_all,
                        dimension_all,
                        radius_all
                ]).T
            )

        if SIMCAP:
            # currently unavailable
            asim0, P, Nc0, N_vec, p_vec = manifold_simcap_analysis(X, 21, seed=SEED)
            df['asim0'] = asim0
            df['P'] = P
            df['Nc0'] = Nc0

        df['mean_cap'] = 1/np.mean(1/capacity_all)
        df['center_corr'] = center_correlation
        df['K'] = K
        df['EVD90'] = D_explained_variance
        df['PR'] = D_participation_ratio
        df['P'] = P
        df['M'] = M
        df['N'] = X[0].shape[0]

        # network details
        df['layer'] = layer
        df['seed'] = seed

        dfs.append(df)

    return pd.concat(dfs)

def process_features(X, P, M, N, NORMALIZE=False, seed=0, verbose=False):
    original_shape = X.shape
    X = X.reshape(P*M, -1)
    if X.shape[-1] > N:
        X = random_projection(X, N, seed=seed)

    if NORMALIZE:
        X = normalize(X)

    # convert to (P, M, N), assuming we have 50 classes, with examples in order
    X = X.reshape(P, M, -1)
    new_shape = X.shape

    if verbose:
        print(f'Original X shape: {original_shape}; new shape {new_shape}')

    # convert to [(N,M1)...(N,Mp)]
    X = [manifold.T for manifold in X]
    return X

def random_projection(X, N_cur, seed=0):
    np.random.seed(seed)
    N = X.shape[1]  # original feature #
    W = np.random.randn(N, N_cur) # randn([pix # x neuron #])
    W = W/np.tile(np.sqrt((W**2).sum(axis=0)), [N,1]) # normalize columns of W
    return np.dot(X,W) # project stimuli onto W

def normalize(X):
    # expects X shaped # stimuli x num features and returns unit wise normalized features.
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    return X

def plot_layerwise(df, measures, eps, manifold_type):
    fig, axes = plt.subplots(nrows=len(measures), ncols=1, figsize=(12,4*len(measures)))

    for ax, measure in zip(axes, measures):
        # filter the df for data of interest
        data = df[
            (df['eps'].apply(lambda x : np.isclose(x, eps)))&
            (df['manifold_type']==manifold_type)
        ]

        # average over seeds / layers / models
        data = data.groupby(['model', 'layer', 'seed']).mean().sort_values(by=['layer'])

        ax = sns.lineplot(
            x='layer',
            y=measure,
            hue='model',
            ax=ax,
            ci='sd',
            data=data,
        )

        sns.despine()

    plt.show()
