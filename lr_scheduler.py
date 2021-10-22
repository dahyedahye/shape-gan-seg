import numpy as np
import matplotlib.pyplot as plt

# added
def sigmoid_lr(lr_initial=0.1, lr_final=0.001, numberEpoch=200, alpha=10, beta=0):

    numberEpoch = int(numberEpoch)
    _index = np.linspace(-1, 1, numberEpoch)
    _sigmoid= 1 / (1 + np.exp(alpha * _index + beta))

    val_initial = _sigmoid[0]
    val_final= _sigmoid[-1]

    a = (lr_initial - lr_final) / (val_initial - val_final)
    b = lr_initial - a * val_initial 

    return a * _sigmoid + b

class _scheduler_learning_rate(object):
    
	def __init__(self, optimizer, epoch=-1):

		self.optimizer 	= optimizer
		self.epoch 		= epoch	

	def step(self, epoch=None):
	
		if epoch is None:
            
			epoch = self.epoch + 1
		
		self.epoch	= epoch
		lr 			= self.get_lr()

		for param_group in self.optimizer.param_groups:
        
			param_group['lr'] = lr

class scheduler_learning_rate_sigmoid(_scheduler_learning_rate): 

	def __init__(self, optimizer, lr_initial, lr_final, numberEpoch, alpha=10, beta=0, epoch=-1):

		# modified
		self.schedule= sigmoid_lr(lr_initial, lr_final, numberEpoch, alpha=alpha, beta=beta)
		self.numberEpoch= numberEpoch

		super(scheduler_learning_rate_sigmoid, self).__init__(optimizer, epoch)

	def get_lr(self, epoch=None):

		if epoch is None:

			epoch = self.epoch

		lr = self.schedule[epoch]

		return lr

	def plot(self):

		fig 	= plt.figure()
		ax		= fig.add_subplot(111)	
		
		ax.plot(self.schedule)
		
		plt.xlim(0, self.numberEpoch + 1)
		plt.xlabel('epoch')
		#plt.ylabel('learning rate')
		plt.grid(linestyle='dotted')
		plt.tight_layout()
		plt.show()

        
# added
class scheduler_learning_rate_sigmoid_double(_scheduler_learning_rate): 

    def __init__(self, optimizer, lr_initial=0.01, lr_top=0.05, lr_final=0.001, numberEpoch=200, ratio=0.25, alpha=10, beta=0, epoch=-1):

        nEpochA = numberEpoch * ratio
        nEpochB = numberEpoch * ratio * 2
        nEpochC = numberEpoch - nEpochA - nEpochB
#         nEpochA = numberEpoch * ratio * 2
#         nEpochB = numberEpoch * ratio * 3
#         nEpochC = numberEpoch - nEpochB
        schedule_A = sigmoid_lr(lr_initial, lr_top, nEpochA + 1, alpha=alpha, beta=beta)  #+1: initial 
        schedule_B = sigmoid_lr(lr_top,   lr_final, nEpochB,     alpha=alpha, beta=beta)
        schedule_C = sigmoid_lr(lr_final,   lr_final, nEpochC,     alpha=alpha, beta=beta)
        self.schedule= np.concatenate([schedule_A,schedule_B,schedule_C])
        self.numberEpoch= numberEpoch

        
#         nEpochA = numberEpoch * ratio * 2
#         nEpochB = numberEpoch - nEpochA
#         schedule_A = sigmoid_lr(lr_initial, lr_top, nEpochA + 1, alpha=alpha, beta=beta)  #+1: initial 
#         schedule_B = sigmoid_lr(lr_top,   lr_final, nEpochB,     alpha=alpha, beta=beta)
#         self.schedule= np.concatenate([schedule_A,schedule_B])
#         self.numberEpoch= numberEpoch

        
        super(scheduler_learning_rate_sigmoid_double, self).__init__(optimizer, epoch)

    def get_lr(self, epoch=None):

        if epoch is None:

            epoch = self.epoch

            lr = self.schedule[epoch]

        return lr

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.schedule)
        plt.xlim(0, self.numberEpoch + 1)
        plt.xlabel('epoch')
        plt.grid(linestyle='dotted')
        plt.tight_layout()
        plt.show()