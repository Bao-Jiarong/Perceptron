'''
 *  Author : Bao Jiarong
 *  Contact: bao.salirong@gmail.com
 *  Created  On: 2020-06-04
 *  Modified On: 2020-06-04
 '''
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.sequential as sequential

def load_data(filename, shuffle = True, split_ratio = 0.8):
    # Load Data
    df = pd.read_csv(filename,sep=',',header=0)
    data = df.values

    # Shuffle Data
    if shuffle == True:
        np.random.shuffle(data)

    # Split Data
    labels = data[:,-1]
    data   = np.delete(data,-1,axis=1)
    n      = int(data.shape[0]*split_ratio)

    return data[:n], labels[:n], data[n:], labels[n:]

def plot(data,label,w,b,plot_loss = False):
    if plot_loss == True:
        fig= plt.figure()
        ax = fig.add_axes([0.1,0.1,0.85,0.85])
        ax.grid(color='b', ls = '-.', lw = 0.25)
        ax.set_xlabel("x axis")
        ax.set_ylabel("y axis")
        d = np.arange(0,history["epoch"])
        ax.plot(d,history["loss"],label="t_loss")
        ax.plot(d,history["val_loss"],label="v_loss")
        ax.legend(loc="upper right",fontsize="x-small")
        plt.show()

    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.85,0.85])
    ax.grid(color='b', ls = '-.', lw = 0.25)
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_xlim(-12,15)
    ax.set_ylim(-12,15)
    x = np.arange(-12,15)
    y = np.arange(-12,15)
    ax.plot(x,0*x)
    ax.plot(0*y,y)

    indice1 = np.argwhere(label == -1)
    indice2 = np.argwhere(label == 1)
    ax.scatter(data[:,0][indice1], data[:,1][indice1], color='b')
    ax.scatter(data[:,0][indice2], data[:,1][indice2], color='r')

    y = -(b+w[0]*x)/w[1]
    ax.plot(x,y)
    plt.show()


model = sequential.Sequential(activation = np.tanh,
                              n_inputs   = 2,
                              initializer= "uniform")

model.compile(loss     = "CE",
              optimizer= "pegasos",
              lr_rate  = 0.01)
#--------------------------------------------------
if sys.argv[1] == "train":
    outs = load_data(filename = "./data/data1_4.csv",
                      shuffle = True,
                      split_ratio = 0.6)
    tr_data  = outs[0]
    tr_label = outs[1]
    val_data = outs[2]
    val_label= outs[3]

    history  = model.fit(tr_data,
                         tr_label,
                         val_data,
                         val_label,
                         epochs    = 300,
                         batch_size= 30)

    loss, acc = model.evaluate(val_data, val_label)

    print("Evaluation, loss =",loss, "acc =",acc,"%")
    model.save_weights("./model/model.txt")

    plot(tr_data,tr_label,history["w"],history["b"],plot_loss = True)

elif sys.argv[1] == "predict":
    w, b = model.load_weights("./model/model.txt")

    x = np.array([float(sys.argv[2]), float(sys.argv[3])])
    y = model.predict(x)
    print("y = ",y)

elif sys.argv[1] == "plot":
    w, b = model.load_weights("./model/model.txt")
    outs = load_data(filename    = "./data/data1_4.csv",
                     shuffle     = True,
                     split_ratio = 0.6)
    val_data  = outs[2]
    val_label = outs[3]
    plot(val_data, val_label, w, b)
