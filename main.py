from torchvision.transforms import Compose, ToTensor, Lambda
import sigtorch as st
import numpy as np
import torch
from scipy.signal import chirp
from scipy import signal

if __name__ == "__main__":
    """
    Playground creating, training, and testing models with arbitrary sized datasets
    """
    mode = 'cnn'

    if mode=='cnn':
        task = 'classification'  # classification or regression

        N = 1000  # number of samples (i.e., 100 images or 100 strings or 100 signals)
        NC = 1  # number of channels
        W = 1000  #  width of sample (e.g., 1000 frames or time steps)
        H = 1  # height of sample (e.g., 24 pixels)
        transform = Compose([st.spectrogram()])
        epochs = 15
        val_split = 0.2
        depth = 152
        model_name = 'CNN'

        y = torch.randint(low=0, high=2, size=(N,))  # N, values in range 0 to n_class
        t = np.linspace(-1, 1, W, endpoint=False)
        x = []
        for label in y:
            if label==0:
                # generate a gaussian pulse
                i, q, e = signal.gausspulse(t, fc=5, retquad=True, retenv=True)
                x.append(q)
            elif label==1:
                # generate a chirp
                w = chirp(t, f0=6, f1=1, t1=0.1, method='linear')
                x.append(w)
        x = torch.from_numpy(np.array(x)).view(N,NC,W,H).float()
        if task == 'regression':
            y = torch.squeeze(torch.hstack((torch.mean(x, dim=2), torch.var(x, dim=2)))) # N, 2
            task_path = 'Mean_and_Var'
        elif task == 'classification':
            task_path = 'Pulse_or_Chirp'
        model = st.CNN()
        model.train(x, y, epochs=epochs, val_split=val_split, task_path=task_path, model_name=model_name, task=task, transform=transform, adv_train='pgd')
        predictions, metric = model.test(x, y)
        print(y)
        print(predictions)
    elif mode=='oc':
        task_path = 'Chirp_or_not'
        model_name = 'SVDD'
        transform = Compose([st.standardize(), st.AWGN()])
        N = 1000  # number of samples (i.e., 100 images or 100 strings or 100 signals)
        NC = 1  # number of channels
        W = 1024  #  width of sample (e.g., 1000 frames or time steps)
        H = 1  # height of sample (e.g., 24 pixels)
        y = torch.zeros(N)  # y=1 is an anomaly, y=0 is in-class. We only train with in-class
        t = np.linspace(-1, 1, W, endpoint=False)
        x = []
        for label in y:
            w = chirp(t, f0=6, f1=1, t1=0.1, method='linear')
            x.append(w)
        x = torch.from_numpy(np.array(x)).view(N,NC,W,H).float()
        epochs = 30

        deep_SVDD = st.OneClass()
        deep_SVDD.train(x, y, n_epochs = epochs, task_path=task_path, model_name=model_name, transform=transform)

        # create testing data
        yt = torch.randint(low=0, high=2, size=(N,))  # Now we test with in-class and out-class (anomaly) data
        xt = []
        for label in yt:
            if label==1:
                # generate a gaussian pulse
                i, q, e = signal.gausspulse(t, fc=5, retquad=True, retenv=True)
                xt.append(q)
            elif label==0:
                # generate a chirp
                w = chirp(t, f0=6, f1=1, t1=0.1, method='linear')
                xt.append(w)
        xt = torch.from_numpy(np.array(xt)).view(N,NC,W,H).float()
        predictions, metric = deep_SVDD.test(xt,yt)
        print(yt)
        print(predictions)
