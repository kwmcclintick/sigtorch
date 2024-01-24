                  .:.
                .:++.                   ███████╗██╗ ██████╗████████╗ ██████╗ ██████╗  ██████╗██╗  ██╗
              .:+++=.   .               ██╔════╝██║██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝██║  ██║
            ..=+++.   -++-              ███████╗██║██║  ███╗  ██║   ██║   ██║██████╔╝██║     ███████║
           .-+++:.    ..:.              ╚════██║██║██║   ██║  ██║   ██║   ██║██╔══██╗██║     ██╔══██║
          :+++-.          .=+:.         ███████║██║╚██████╔╝  ██║   ╚██████╔╝██║  ██║╚██████╗██║  ██║
        .:+++.            .=++-.        ╚══════╝╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
        :+++.              .+++:
       .-++.                .++=.       .
       .-++.                .++=.                           ......                             .
       .-++-                :++-.       .                  ..     ..            ..: .       ..  ...
        .+++.              .+++.              .           .         . ... .. .:..     .            ..
         .+++-            -+++.         :.   ....      ....          .               .  ..          .
          .=+++-..    ..-+++=.              ...   ..... .
           ..++++++++++++++..              .         ..
              ..-======-..

# CNN M-Class Classification

         ██████╗███╗   ██╗███╗   ██╗
        ██╔════╝████╗  ██║████╗  ██║
        ██║     ██╔██╗ ██║██╔██╗ ██║
        ██║     ██║╚██╗██║██║╚██╗██║
        ╚██████╗██║ ╚████║██║ ╚████║
         ╚═════╝╚═╝  ╚═══╝╚═╝  ╚═══╝
                                                  -*+**************+:..                               #+++++++++=-.       ..............                 :.:.
                                                  -*+-................:                               #=**********=:.     :*+++++++-                     :-=.    HORSE
                                                  -*+-.+****************+         :*************.     #=*=+********#=:    :**=*******-.                  :.:.
        .=*-+:            .-.                    -*+-.=+*===============--       :**..............   #=*=+#**********-.  :**==*++++++=:.                :-=.
        :==#.:+.     .=*--+..=.    ....          -*+-.=++.:===============--.    :**.*++++++++++++*..#=*=+#***********+:.:**==*=*******+..
        .=. --.+.    .=..--  .+..++....:*..      -*+-.=++.:*+*+++++++++++++*:..  :**.**-............:.-*=+#***-++++++++++.:-==*=+=#======--.             .       DOG
          .=+.  .=*-.#.   .*-*=.       . .=.     -*+-.=++.:**=................+  :**.**-:*+***********-.:+#***-*=========== .=*=++#=+++++++=:.           .
             -..     -:.   ..+:..    .:#..-=:..  -*+-.=++.:**=.+++++++++++++++++*:**.**-:*#.::::::::::::::+***-*=*=----------..:++#=-*+-------:          .
              -..    .==.     .:+         :++:   -*+-.=++.:**=.++*:::::::::::::::::*.**-:*#.=+***********#::+*-*=***----------:..+#=-*++--------:.       .
              .--.    ..+----=-.         .=.       .:.=++.:**=.++*................:-.**-:*#.=*=..............-:*=***=+=++++++++=:.:=-*++=*=======+:      .       CAT
               .+.                      =:            =++.:**=.++*................: .**-:*#.=*=....:::::....  .+=***=+-**********-. -*++=*+=======++.   .:..
                ..=:..                .+.             ..:.:**=.++*.........=--=::.: ...::*#.=*=...:=--:..::.   .:***=+-***########*-..-+=*+=++......::  :-=.
                    .:=-.           ..+:       ......   .::**=.++*.........=--=...:.....:*#.=*=....:::::::........+*=+-**+:..----:.:.   -*+=++.......:  :.:.    ------
                     ...::++++*++-:...         ............++=.++*................:.......=-+*+:::..................=*-**+:..=--=..:......+=++.......:  :-=.====|BIRD|
                       .-+=...=.                           ..=:***:...............:        .=*=.............        ..=#*+:........::.......*+:-:....:  ...     ------
                       ..+..=#.
                             ..
                             
This example demonstrates a dataset comprised of gaussian pulse and chirp signals.

![alt text](https://llcad-github.llan.ll.mit.edu/kwmcclintick/SigTorch/blob/main/chirp.png)
![alt text](https://llcad-github.llan.ll.mit.edu/kwmcclintick/SigTorch/blob/main/pulse.png)

To demonstrate classification capabilities, we estimate one unknown parameters for every sample observed: if the sample is from the gaussian pulse class or the chirp class. The following excerpt from main.py is run:
```
import sigtorch as st
import numpy as np
import torch
from scipy.signal import chirp
from scipy import signal

N = 1000  # number of samples (i.e., 100 images or 100 strings or 100 signals)
NC = 1  # number of channels
W = 1000  #  width of sample (e.g., 1000 frames or time steps)
H = 1  # height of sample (e.g., 24 pixels)
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
task_path = 'Pulse_or_Chirp'
model = st.CNN()
model.train(x, y, epochs=epochs, val_split=val_split, task_path=task_path, model_name=model_name, task=task)
predictions, metric = model.test(x, y)
print(y)
print(predictions)
```
Returning the following stdout:
```
2023-10-02 08:12:41,662 [INFO] [sigtorch.py:540]: Training on inputs of shape: 1000 samples, 1 channels/sample, 1000 sample width, 1 sample height
2023-10-02 08:12:41,662 [INFO] [sigtorch.py:541]: Training on outputs of shape: torch.Size([1000])
Training, Epoch: [0 / 15], CrossEntropyLoss()=0.282 [========================================] 50/50 [100%] in 1.6s (28.16/s)
Validation, Epoch: [0 / 15], CrossEntropyLoss()=0.000 <●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●> 13/13 [100%] in 0.1s (129.38/s)
2023-10-02 08:12:44,301 [WARNING] [sigtorch.py:626]: Validation loss is less than training loss!
Training, Epoch: [1 / 15], CrossEntropyLoss()=0.000 [========================================] 50/50 [100%] in 0.7s (70.50/s)
...
Training, Epoch: [15 / 15], CrossEntropyLoss()=0.000 [========================================] 50/50 [100%] in 0.7s (70.64/s)
Validation, Epoch: [15 / 15], CrossEntropyLoss()=0.000 <●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●> 13/13 [100%] in 0.1s (129.35/s)
2023-10-02 08:12:55,220 [WARNING] [sigtorch.py:626]: Validation loss is less than training loss!
2023-10-02 08:12:55,220 [INFO] [sigtorch.py:648]: Starting testing...
2023-10-02 08:12:57,307 [INFO] [sigtorch.py:669]: Testing time: 2.087
2023-10-02 08:12:57,320 [INFO] [sigtorch.py:676]: Test set F1 score: 1.00000%
2023-10-02 08:12:57,321 [INFO] [sigtorch.py:684]: Finished testing
tensor([0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1,
        ...
        0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0,
        0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0])
[0 0 0 1 0 1 0 0 0 1 0 0 0 1 0 1 0 0 1 0 1 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0
 1 1 1 0 1 0 1 0 0 1 1 0 1 1 1 0 0 1 0 0 1 1 0 1 0 0 0 0 1 1 0 1 0 0 0 0 1
 ...
 1 1 0 1 0 0 0 0 1 1 1 1 1 1 0 1 1 1 0 1 0 1 1 1 0 0 1 1 1 1 1 1 1 1 0 0 1
 0 1 1 1 1 1 0 0 0 0 0 0 0 1 0 0 0 1 0 0 1 0 0 1 0 0 1 1 1 1 0 0 1 1 1 0 0
 0]
```


# CNN Regression
This example demonstrates a dataset comprised of gaussian pulse and chirp signals. To demonstrate regression capabilities, we estimate two unknown parameters for every sample observed: mean and variance. The following excerpt from main.py is run:
```
N = 1000  # number of samples (i.e., 100 images or 100 strings or 100 signals)
NC = 1  # number of channels
W = 1000  #  width of sample (e.g., 1000 frames or time steps)
H = 1  # height of sample (e.g., 24 pixels)
epochs = 100
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
y = torch.squeeze(torch.hstack((torch.mean(x, dim=2), torch.var(x, dim=2)))) # N, 2
task_path = 'Mean_and_Var'
model = st.CNN()
model.train(x, y, epochs=epochs, val_split=val_split, task_path=task_path, model_name=model_name, task=task)
predictions, metric = model.test(x, y)
print(y)
print(predictions)
```
Returning the following stdout:
```
2023-09-29 15:01:53,291 [INFO] [sigtorch.py:538]: Training on inputs of shape: 1000 samples, 1 channels/sample, 1000 sample width, 1 sample height
2023-09-29 15:01:53,291 [INFO] [sigtorch.py:539]: Training on outputs of shape: torch.Size([1000, 2])
Training, Epoch: [0 / 100], MSELoss()=0.119 [========================================] 50/50 [100%] in 10.7s (4.68/s)
Validation, Epoch: [0 / 100], MSELoss()=0.001 <●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●> 13/13 [100%] in 0.1s (125.24/s)
2023-09-29 15:02:10,861 [WARNING] [sigtorch.py:624]: Validation loss is less than training loss!

...

Validation, Epoch: [100 / 100], MSELoss()=0.000 <●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●> 13/13 [100%] in 0.1s (207.59/s)
2023-09-29 15:03:21,970 [INFO] [sigtorch.py:229]: Best checkpoint yet, saving with name CNN to Mean_and_Var/ckpt.json and Mean_and_Var/ckpt.pt
2023-09-29 15:03:22,043 [WARNING] [sigtorch.py:624]: Validation loss is less than training loss!
2023-09-29 15:03:22,045 [INFO] [sigtorch.py:646]: Starting testing...
2023-09-29 15:03:24,163 [INFO] [sigtorch.py:667]: Testing time: 2.118
2023-09-29 15:03:24,164 [INFO] [sigtorch.py:677]: Test set MSE: 0.00001
2023-09-29 15:03:24,164 [INFO] [sigtorch.py:682]: Finished testing
[[ 1.2219e-10,  6.6381e-02],
        [-2.2268e-03,  5.0049e-01],
        [ 1.2219e-10,  6.6381e-02],
        ...,
        [ 1.2219e-10,  6.6381e-02],
        [ 1.2219e-10,  6.6381e-02],
        [ 1.2219e-10,  6.6381e-02]]
[[-0.00166208  0.06314141]
 [-0.00219272  0.49549511]
 [-0.00166208  0.06314141]
 ...
 [-0.00166208  0.06314141]
 [-0.00166208  0.06314141]
 [-0.00166208  0.06314141]]
```


# SVDD One-Class Anomaly Detection

         ██████╗ ███╗   ██╗███████╗     ██████╗██╗      █████╗ ███████╗███████╗
        ██╔═══██╗████╗  ██║██╔════╝    ██╔════╝██║     ██╔══██╗██╔════╝██╔════╝
        ██║   ██║██╔██╗ ██║█████╗█████╗██║     ██║     ███████║███████╗███████╗
        ██║   ██║██║╚██╗██║██╔══╝╚════╝██║     ██║     ██╔══██║╚════██║╚════██║
        ╚██████╔╝██║ ╚████║███████╗    ╚██████╗███████╗██║  ██║███████║███████║
         ╚═════╝ ╚═╝  ╚═══╝╚══════╝     ╚═════╝╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝
                                        :#%+         +%%:                             XXXX
                XXXX          .=%-                           .#*.                     XXXX
                XXXX        ..                      ...            :.                 XXXX
                XXXX      :#-                    .=*-=*.          .*-
                        ...                       .#.  +-             .-.
                       .=+.         .=*+:          .+**-               :#.
                        .:++=.     .#: .+=                                .
                     .#.:+. .*     .=+:-*....                     ....   .+=.
                    .-. .*-.++       ...:*:.=*.                 .*-.:*:   .:.
                    .     ...           -+  :#.:#::#:           .#. .*-     :.
                   -+                    .=+:  +.  .+     :**:   .:+=.      :%.
                             .=##:.            .+##+.    *-  =+.    ..::..
                  ..         *.  --                      -+..*-     :+. -+.:--*.
                  +:         =+:-#.         ...                     :+..-+#. .*:
                                           =*+*=                     .-=..*:..*:
                 .-.             .:***:   .*  .*    -#**:.                 .-.*-
        XXXX     .*:             .*  .+    -*+*-    +.  #.  :=+-.            .*:
        XXXX                     .+*=++             ++=*=. -=. .+
        XXXX      .+               ...               ...   :*:.+=            .+.  XXXX
                  .#:                           :*::*-  **+#....             +-   XXXX
                                  .*#*.    =*-*=*-  -*.==. .#.                    XXXX
                    +=           .*  .+   .*   * -##:  .+#*#.               #.
                    .=.          .++.=+    -#*#-                    .+##*. ..
                      ..                        ...                 =-  .*:.               XXXX
                      .+=.                    .*=:*-.               .*=-*#:                XXXX
                                              =+  .#.                                      XXXX
                         .*+.                  -**+.                 :#-.
                             .+#:                        ...    .+#:              XXXX
                               .. :-                   :*..=+.=. ..               XXXX
                                  ..:. ..              =+-.:#..                   XXXX
                                       ...  :%%+  #%#. ..=+:.

This example demonstrates a dataset comprised of gaussian pulse and chirp signals. To demonstrate one-class classification capabilities, we estimate one unknown parameters for every sample observed: if the sample is from the chirp class, or if it is not from the chirp class. To be clear, this problem is different from a binary classification problem because training is done with only one class of samples. The following excerpt from main.py is run:
```
task_path = 'Chirp_or_not'
model_name = 'SVDD'
N = 1000  # number of samples (i.e., 100 images or 100 strings or 100 signals)
NC = 1  # number of channels
W = 1024  #  width of sample (e.g., 1000 frames or time steps)
H = 1  # height of sample (e.g., 24 pixels)
y = torch.zeros(N)  # y=0 is an anomaly, y=1 is in-class. We only train with in-class
t = np.linspace(-1, 1, W, endpoint=False)
x = []
for label in y:
    w = chirp(t, f0=6, f1=1, t1=0.1, method='linear')
    x.append(w)
x = torch.from_numpy(np.array(x)).view(N,NC,W,H).float()
epochs = 30

deep_SVDD = st.OneClass()
deep_SVDD.train(x, y, n_epochs = epochs, task_path=task_path, model_name=model_name)

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
```
Returning the following stdout:
```
2023-10-02 08:22:19,820 [INFO] [sigtorch.py:361]: Training on inputs of shape: 1000 samples, 1 channels/sample, 1024 sample width, 1 sample height
2023-10-02 08:22:19,820 [INFO] [sigtorch.py:362]: Training on outputs of shape: 1000 samples in range, where y=0 are anomalies and y=1 are in-class
2023-10-02 08:22:19,820 [INFO] [sigtorch.py:364]: 1D mode enabled
Training, Epoch: [0 / 31], Quadratic Loss=870.735 [========================================] 8/8 [100%] in 0.2s (43.08/s)
Training, Epoch: [1 / 31], Quadratic Loss=150.402 [========================================] 8/8 [100%] in 0.1s (79.60/s)
Training, Epoch: [2 / 31], Quadratic Loss=63.504 [========================================] 8/8 [100%] in 0.1s (79.69/s)
...
Training, Epoch: [28 / 31], Quadratic Loss=0.000 [========================================] 8/8 [100%] in 0.1s (79.63/s)
Training, Epoch: [29 / 31], Quadratic Loss=0.000 [========================================] 8/8 [100%] in 0.1s (79.67/s)
Training, Epoch: [30 / 31], Quadratic Loss=0.000 [========================================] 8/8 [100%] in 0.1s (79.67/s)
2023-10-02 08:22:24,847 [INFO] [sigtorch.py:449]: Starting testing...
2023-10-02 08:22:24,923 [INFO] [sigtorch.py:472]: Testing time: 0.076
2023-10-02 08:22:24,925 [INFO] [sigtorch.py:479]: Test set AUC: 100.00%
2023-10-02 08:22:24,925 [INFO] [sigtorch.py:484]: Finished testing.
tensor([0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1,
        1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1,
        ...
        1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0,
        0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
[4.71457817e-10 2.09047198e+00 2.09047198e+00 2.09047198e+00
 4.71457817e-10 4.71457817e-10 2.09047198e+00 2.09047198e+00
 ...
 2.09047198e+00 4.71457817e-10 2.09047198e+00 4.71457817e-10
 2.09047198e+00 2.09047198e+00 4.71457817e-10 4.71457817e-10]
```
