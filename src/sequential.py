'''
 *  Author : Bao Jiarong
 *  Contact: bao.salirong@gmail.com
 *  Created  On: 2020-06-04
 *  Modified On: 2020-06-04
 '''
from .algo import *

class Sequential(Algo):
    #--------------------------------------------------------
    # Constructor
    #---------------------------------------------------------
    def __init__(self,activation,n_inputs,initializer="normal"):
        Algo.__init__(self,activation,n_inputs,initializer)

    def compile(self,loss,optimizer="SGD",lr_rate=0.01):
        if loss == "MSE":
            self.loss = self.MSE
        else:
            self.loss = self.CE
        self.acc      = self.accuracy
        self.lr_rate  = lr_rate
        self.optimizer= optimizer

    # LOSS FUNCTIONS.......................
    # cross entropy loss function
    def CE(self,X,Y):
        L = len(X)
        d = 0
        for i in range(L):
            x  = X[i]   # real (x1,x2)
            y  = Y[i]   # class -1 or 1
            y_ = self.f(self.w @ x + self.b)
            d  = d -y*np.log(sigmoid(y_)) - (1-y)*np.log(1-sigmoid(y_))
        return d/L

    # mean square error
    def MSE(self,X,Y):
        L = len(X)
        d = 0
        for i in range(L):
            x  = X[i]   # real (x1,x2)
            y  = Y[i]   # class -1 or 1
            y_ = self.f(self.w @ x + self.b)
            d  = d + abs(y - y_)
        return d/L

    # accuracy functions...................
    def accuracy(self,X,Y):
        L = len(X)
        a = 0
        for i in range(L):
            x  = X[i]   # real (x1,x2)
            y  = Y[i]   # class -1 or 1
            y_ = np.sign(self.f(self.w @ x + self.b))
            if y == y_:
                a += 1
        return a*100/L

    def evaluate(self,valid_data,valid_label):
        loss = self.MSE(valid_data,valid_label)
        acc = self.accuracy(valid_data,valid_label)
        return loss,acc

    def fit(self,train_x,train_y,valid_x,valid_y,epochs,batch_size,lamda = 0.1):
        L      = len(train_x)
        self.w = np.random.uniform(0,1,train_x.shape[-1])
        self.b = random.random()
        err    = []
        M      = int(L/batch_size)
        history= {"loss"    :[],
                  "acc"     :[],
                  "val_loss":[],
                  "val_acc" :[],
                  "w"       :self.w,
                  "b"       :self.b,
                  "epoch"   :epochs}
        t = 0
        for epoch in range(epochs):
            for m in range(M):
                # SGD Optimizer (Stochastic Gradient Descent)
                if self.optimizer == "SGD":
                    d = 0
                    x = np.zeros(train_x.shape[-1])
                    for j in range(batch_size):
                        i = np.random.randint(L)
                        train_y_= self.f(self.w@train_x[i]+self.b)
                        d += (train_y[i] - train_y_)
                        x += train_x[i]
                    x /= batch_size
                    d /= batch_size
                    self.w = self.w + self.lr_rate * d * x
                    self.b = self.b + self.lr_rate * d
                # Pegasos Optimizer
                else:
                    for j in range(batch_size):
                        t += 1
                        i  = np.random.randint(L)
                        x  = train_x[i]
                        y  = train_y[i]
                        score= self.w @ x + self.b
                        zeta = 1.0 / (t * lamda)
                        if y * score < 1:
                            self.w = (1 - zeta * lamda) * self.w + (zeta * y) * x
                            self.b = (1 - zeta * lamda) * self.b + (zeta * y)
                        else:
                            self.w = (1 - zeta * lamda) * self.w
                            self.b = (1 - zeta * lamda) * self.b

            tr_loss = round(self.loss(train_x,train_y),4)
            tr_acc  = round(self.acc(train_x ,train_y),4)
            va_loss = round(self.loss(valid_x,valid_y),4)
            va_acc  = round(self.acc(valid_x ,valid_y),4)

            history["loss"    ].append(tr_loss)
            history["acc"     ].append(tr_acc )
            history["val_loss"].append(va_loss)
            history["val_acc" ].append(va_loss)

            if epoch % 10 == 0:
                msg = "epoch = {:4} \ttr_loss = {:2.4} \ttr_acc = {:2.4} " + \
                      "\tval_loss = {:2.4} \tval_acc = {:2.4}"
                print(msg.format(epoch,tr_loss,tr_acc,va_loss,va_acc))
        history["w"] = self.w
        history["b"] = self.b

        return history

    def save_weights(self,filename):
        f = open(filename, "w")
        f.write(str(self.w[0])+" "+str(self.w[1])+" "+str(self.b))
        f.close()

    def load_weights(self,filename):
        f = open(filename, "r")
        a = f.read()
        f.close()

        a = [float(x) for x in a.split(" ")]
        self.w = np.array(a[:2])
        self.b = a[2]
        return self.w,self.b

    def predict(self,x):
        return np.sign(self.f(self.w@x+self.b))
