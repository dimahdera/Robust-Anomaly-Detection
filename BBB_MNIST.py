import tensorflow as tf
from tensorflow import keras
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import time, sys
import pickle
import timeit
import wandb
import xlsxwriter
import pandas as pd

os.environ["WANDB_API_KEY"] = "0cc757590f04bfdf0e3c023a3351d6ee061851c4"
plt.ioff()
mnist = tf.keras.datasets.mnist
fmnist = tf.keras.datasets.fashion_mnist
# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def log_gaussian(x, mu, sigma):
    return -0.5 * np.log(2 * np.pi) - tf.math.log(tf.abs(sigma)) - (x - mu) ** 2 / (2 * sigma ** 2)


def get_kl_divergence(mu, sigma):
    """
    Compute KL divergence between posterior and prior.
    log(q(theta)) - log(p(theta)) where
    p(theta) = pi*N(0,sigma1) + (1-pi)*N(0,sigma2)     
    mu = the mu variable used when sampling
    sigma= the sigma variable used when sampling
    prior = the prior object with parameters
    sample = the sample from the posterior
    """
    pi_mix = 0.25
    log_sigma1 = -1.
    log_sigma2 = -7.0

    sigma1 = tf.exp(log_sigma1)
    sigma2 = tf.exp(log_sigma2)

    # Flatten to a vector
    # sample = tf.reshape(sample, [-1])
    mu = tf.reshape(mu, [-1])
    sigma = tf.reshape(sigma, [-1])
    epsilon = tf.random.normal(sigma.shape, 0.0, 0.1, sigma.dtype)
    sample = mu + (tf.nn.softplus(sigma)) * epsilon
    # Get the log probability distribution of your sampled variable
    # So essentially get: q( theta | mu, sigma )   
    posterior = log_gaussian(sample, mu, sigma)  # Normal(mu, sigma)    
    prior_1 = log_gaussian(sample, 0.0, sigma1)  # Normal(0.0, sigma1)
    prior_2 = log_gaussian(sample, 0.0, sigma2)  # Normal(0.0, sigma2)

    # get: sum( log[ q( theta | mu, sigma ) ] )
    q_theta = tf.reduce_mean(posterior)  # .log_prob(sample))

    # get: sum( log[ p( theta ) ] ) for mixture prior
    #    mix1 = tf.reduce_sum(prior_1.log_prob(sample)) + tf.math.log(pi_mix)
    #    mix2 = tf.reduce_sum(prior_2.log_prob(sample)) + tf.math.log(1.0 - pi_mix)
    mix1 = tf.reduce_mean(prior_1) + tf.math.log(pi_mix)
    mix2 = tf.reduce_mean(prior_2) + tf.math.log(1.0 - pi_mix)
    # Compute KL divergence
    KL = q_theta - tf.reduce_logsumexp([mix1, mix2])
    return KL


class BBB_LinearLayer(keras.layers.Layer):
    """y = w.x + b"""

    def __init__(self, units):
        super(BBB_LinearLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        pi = 0.25
        log_sigma1 = -1.
        log_sigma2 = -7.0
        sigma_one, sigma_two = math.exp(log_sigma1), math.exp(log_sigma2)
        sigma_mix = np.sqrt(pi * np.square(sigma_one) + (1.0 - pi) * np.square(sigma_two))
        rho_max_init = math.log(math.exp(sigma_mix / 2.0) - 1.0)
        rho_min_init = math.log(math.exp(sigma_mix / 4.0) - 1.0)
        self.w1 = self.add_weight(shape=(input_shape[-1], self.units),
                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None), name='w1',
                                  trainable=True)
        #  self.b1 = self.add_weight(shape=(self.units,), initializer=tf.random_normal_initializer( mean=0.0, stddev=0.01, seed=None), name='b1', trainable=True)       
        self.w1_std = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer=tf.random_uniform_initializer(minval=rho_min_init,
                                                                                maxval=rho_max_init, seed=None),
                                      name='wl_std', trainable=True)

    # self.b1_std = self.add_weight(shape=(self.units,), initializer=tf.random_uniform_initializer(minval=rho_min_init, maxval=rho_max_init,seed=None), name='bl_std', trainable=True)        

    def call(self, inl):
        epsilon_w = tf.random.normal(self.w1_std.shape, 0.0, 0.1)  # , self.w1_std.dtype)
        # epsilon_b = tf.random.normal(self.b1_std.shape, 0.0, 0.1)#, self.b1_std.dtype)    
        w1_1 = self.w1 + (tf.nn.softplus(self.w1_std)) * epsilon_w
        # b1_1 = self.b1 + (tf.nn.softplus(self.b1_std) ) * epsilon_b        
        output = tf.matmul(inl, w1_1)  # + b1_1     
        return output


       		
class mysoftmax(keras.layers.Layer):
    """Mysoftmax"""
    def __init__(self):
        super(mysoftmax, self).__init__()
    def call(self, mu_in):
        mu_out = tf.nn.softmax(mu_in)
        return mu_out

class BBBReLU(keras.layers.Layer):
    def __init__(self):
        super(BBBReLU, self).__init__()
    def call(self, mu_in):
        mu_out = tf.nn.relu(mu_in)
        return mu_out

class BBB_model(tf.keras.Model):
  def __init__(self, units,num_classes,  name=None):
    super(BBB_model, self).__init__()
    self.units = units
    self.num_classes = num_classes
    self.linear1 = BBB_LinearLayer(self.units)
    self.bbb_relu = BBBReLU()
    self.linear2 = BBB_LinearLayer(self.num_classes)
    self.mysoftma = mysoftmax()

  def call(self, inputs, training=True):
    output = self.linear1(inputs)
    output = self.bbb_relu(output)
    output = self.linear2(output)
    output = self.mysoftma(output)
    return output


def main_function(input_dim=28, num_unit=800, class_num=10 , batch_size=50, epochs =20, lr=0.001, lr_end = 0.0001,
        Random_noise=False, gaussain_noise_std=0.5, Adversarial_noise=False, epsilon = 0.1,corrupted_images=False,kind="Impulse", adversary_target_cls=3, Testing=False, Testing_F=False,
        Targeted=False, Training = False, continue_training = False,  saved_model_epochs=50):   
       
    PATH = './bbb_saved_models/BBB_epoch_{}/'.format( epochs)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = tf.reshape(x_train, [-1, 28*28])
    x_test = tf.reshape(x_test, [-1, 28*28])
    one_hot_y_train = tf.one_hot(y_train.astype(np.float32), depth=class_num)
    one_hot_y_test = tf.one_hot(y_test.astype(np.float32), depth=class_num)
    tr_dataset = tf.data.Dataset.from_tensor_slices((x_train, one_hot_y_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, one_hot_y_test)).batch(batch_size)
    
    bbb_model = BBB_model(units=num_unit, num_classes=class_num, name = 'bbb')
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False) 
    num_train_steps = epochs * int(x_train.shape[0] /batch_size)
#    step = min(step, decay_steps)
#    ((initial_learning_rate - end_learning_rate) * (1 - step / decay_steps) ^ (power) ) + end_learning_rate
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=lr,  decay_steps=num_train_steps,  end_learning_rate=lr_end, power=2.)   
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)#, clipnorm=1.0)
    kl_factor = 1. / (batch_size)  # *time_step)
    @tf.function  # Make it fast.
    def train_on_batch(x, y):
        with tf.GradientTape() as tape:
            logits = bbb_model(x, training=True)
            bbb_model.trainable = True
          #  tf.print(bbb_model.layers)
         #   print(bbb_model.layers)
          #  tf.print(bbb_model.layers[0].trainable_weights[0])
           # print(bbb_model.layers[1].trainable_weights[1])
            dense_kl1 = get_kl_divergence(bbb_model.layers[0].trainable_weights[0], bbb_model.layers[0].trainable_weights[1])
            dense_kl2 = get_kl_divergence(bbb_model.layers[2].trainable_weights[0],bbb_model.layers[2].trainable_weights[1])
            KL_term = dense_kl1 + dense_kl2
            loss = loss_fn(y, logits) + kl_factor * KL_term

        gradients = tape.gradient(loss, bbb_model.trainable_weights)
     #   for g,v in zip(gradients, bbb_model.trainable_weights):
     #       tf.print(v.name, tf.reduce_max(g)) 
        gradients = [(tf.where(tf.math.is_nan(grad), tf.constant(1.0e-5, shape=grad.shape), grad)) for grad in gradients]
        gradients = [(tf.where(tf.math.is_inf(grad), tf.constant(1.0e-5, shape=grad.shape), grad)) for grad in gradients]
        optimizer.apply_gradients(zip(gradients, bbb_model.trainable_weights))
        return loss, logits, gradients

    @tf.function
    def validation_on_batch(x, y):                     
        logits = bbb_model(x, training=False)
        #bbb_model.trainable = False
        dense_kl1 = get_kl_divergence(bbb_model.layers[0].trainable_weights[0],  bbb_model.layers[0].trainable_weights[1])
        dense_kl2 = get_kl_divergence(bbb_model.layers[2].trainable_weights[0],  bbb_model.layers[2].trainable_weights[1])
        KL_term = dense_kl1 + dense_kl2
        loss = loss_fn(y, logits) + kl_factor * KL_term
        return loss, logits

    @tf.function
    def test_on_batch(x, y):  
        bbb_model.trainable = False
        mu_out = bbb_model(x, training=False)
        return mu_out
    
    @tf.function()
    def create_adversarial_pattern(input_image, input_label):
          with tf.GradientTape() as tape:
            tape.watch(input_image)
            bbb_model.trainable = False
            prediction = bbb_model(input_image)
            loss = loss_fn(y, prediction)
          # Get the gradients of the loss w.r.t to the input image.
          gradient = tape.gradient(loss, input_image)
          # Get the sign of the gradients to create the perturbation
          signed_grad = tf.sign(gradient)
          return signed_grad         
    if Training:
        wandb.init(entity = "dimah", project="BBB_mnist_epochs_{}_lr_{}_latest".format(epochs, lr))
        if continue_training:
            saved_model_path = './bbb_saved_models/bbb_epoch_{}/'.format(saved_model_epochs)
            bbb_model.load_weights(saved_model_path + 'bbb_model')
        train_acc = np.zeros(epochs) 
        valid_acc = np.zeros(epochs)
        train_err = np.zeros(epochs)
        valid_error = np.zeros(epochs)
        
        start = timeit.default_timer()       
        for epoch in range(epochs):
            print('Epoch: ', epoch+1, '/' , epochs)           
            acc1 = 0
            acc_valid1 = 0 
            err1 = 0
            err_valid1 = 0
            tr_no_steps = 0
            va_no_steps = 0           
            #-------------Training--------------------
            for step, (x, y) in enumerate(tr_dataset):                         
                update_progress(step/int(x_train.shape[0]/(batch_size)) )                
                loss, mu_out, gradients = train_on_batch(x, y)                      
                err1+= loss.numpy() 
                corr = tf.equal(tf.math.argmax(mu_out, axis=1),tf.math.argmax(y,axis=1))
                accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))                            
                acc1+=accuracy.numpy()                 
                if step % 50 == 0:
                    print('\n gradient', np.mean(gradients[0].numpy()))
                    print("\n Step:", step, "Loss:" , float(err1/(tr_no_steps + 1.)))
                    print("Total Training accuracy so far: %.3f" % float(acc1/(tr_no_steps + 1.)))                                                                   
                tr_no_steps+=1 
                wandb.log({"Total Training Loss": loss.numpy() ,
                             "Training Accuracy per minibatch": accuracy.numpy() ,                                                 
                             "gradient per minibatch": np.mean(gradients[0]),                              
                             'epoch': epoch
                    })        
            train_acc[epoch] = acc1/tr_no_steps
            train_err[epoch] = err1/tr_no_steps        
            print('Training Acc  ', train_acc[epoch])
            print('Training error  ', train_err[epoch])         
            #---------------Validation----------------------                  
            for step, (x, y) in enumerate(val_dataset):               
                update_progress(step / int(x_test.shape[0] / (batch_size)) )   
                total_vloss, mu_out   = validation_on_batch(x, y)                
                err_valid1+= total_vloss.numpy()                               
                corr = tf.equal(tf.math.argmax(mu_out, axis=-1),tf.math.argmax(y,axis=-1))
                va_accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
                acc_valid1+=va_accuracy.numpy() 
                
                if step % 50 == 0:                   
                    print("Step:", step, "Loss:", float(total_vloss))
                    print("Total validation accuracy so far: %.3f" % va_accuracy)               
                va_no_steps+=1
                wandb.log({"Total Validation Loss": total_vloss.numpy() ,                              
                               "Validation Acuracy per minibatch": va_accuracy.numpy()                                                         
                                })          
            valid_acc[epoch] = acc_valid1/va_no_steps      
            valid_error[epoch] = err_valid1/va_no_steps
            stop = timeit.default_timer()
            bbb_model.save_weights(PATH + 'bbb_model')   
            wandb.log({"Average Training Loss":  train_err[epoch],                        
                        "Average Training Accuracy": train_acc[epoch],                                            
                        "Average Validation Loss": valid_error[epoch],                                           
                        "Average Validation Accuracy": valid_acc[epoch],                                           
                        'epoch': epoch
                       }) 
            print('Total Training Time: ', stop - start)
            print('Training Acc  ', train_acc[epoch])
            print('Validation Acc  ', valid_acc[epoch])           
            print('------------------------------------')
            print('Training error  ', train_err[epoch])
            print('Validation error  ', valid_error[epoch])           
        #-----------------End Training--------------------------             
        bbb_model.save_weights(PATH + 'bbb_model')        
        if (epochs > 1):
            fig = plt.figure(figsize=(15,7))
            plt.plot(train_acc, 'b', label='Training acc')
            plt.plot(valid_acc,'r' , label='Validation acc')
            plt.ylim(0, 1.1)
            plt.title("BBB on MNIST Data")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend(loc='lower right')
            plt.savefig(PATH + 'BBB_on_MNIST_Data_acc.png')
            plt.close(fig)    
    
            fig = plt.figure(figsize=(15,7))
            plt.plot(train_err, 'b', label='Training error')
            plt.plot(valid_error,'r' , label='Validation error')            
            plt.title("BBB on MNIST Data")
            plt.xlabel("Epochs")
            plt.ylabel("Error")
            plt.legend(loc='upper right')
            plt.savefig(PATH + 'BBB_on_MNIST_Data_error.png')
            plt.close(fig)
        
        f = open(PATH + 'training_validation_acc_error.pkl', 'wb')         
        pickle.dump([train_acc, valid_acc, train_err, valid_error], f)                                                   
        f.close()                  
             
        textfile = open(PATH + 'Related_hyperparameters.txt','w')    
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No of units : ' +str(num_unit))
        textfile.write('\n Number of Classes : ' +str(class_num))
        textfile.write('\n No of epochs : ' +str(epochs))
        textfile.write('\n Initial Learning rate : ' +str(lr)) 
        textfile.write('\n Ending Learning rate : ' +str(lr_end))  
        textfile.write('\n batch size : ' +str(batch_size)) 
        textfile.write('\n KL term factor : ' +str(kl_factor))         
        textfile.write("\n---------------------------------")          
        if Training: 
            textfile.write('\n Total run time in sec : ' +str(stop - start))
            if(epochs == 1):
                textfile.write("\n Averaged Training  Accuracy : "+ str( train_acc))
                textfile.write("\n Averaged Validation Accuracy : "+ str(valid_acc ))
                    
                textfile.write("\n Averaged Training  error : "+ str( train_err))
                textfile.write("\n Averaged Validation error : "+ str(valid_error ))
            else:
                textfile.write("\n Averaged Training  Accuracy : "+ str(np.mean(train_acc[epoch])))
                textfile.write("\n Averaged Validation Accuracy : "+ str(np.mean(valid_acc[epoch])))
                
                textfile.write("\n Averaged Training  error : "+ str(np.mean(train_err[epoch])))
                textfile.write("\n Averaged Validation error : "+ str(np.mean(valid_error[epoch])))
        textfile.write("\n---------------------------------")                
        textfile.write("\n---------------------------------")    
        textfile.close()
    #-------------------------Testing-----------------------------    
    if (Testing_F):
        test_path = '/updated/test_results_FASHION/'
        if Random_noise:
            test_path = 'test_results_random_noise_{}/'.format(gaussain_noise_var)                     
        if not os.path.exists(PATH+ test_path ):
            os.makedirs(PATH+ test_path )
        (x_train, y_train), (x_test, y_test) = fmnist.load_data()
        x_train, x_test = x_train / 255.0, x_test /255.0
       # x_train = np.reshape(x_train, [-1, 28*28])
        x_test = np.reshape(x_test, [-1, 28*28])
        x_test = x_test.astype('float32')
       # x_test = tf.expand_dims(x_test, -1)
        val_dataset = tf.data.Dataset.from_tensor_slices((x_test, one_hot_y_test)).batch(batch_size)
        bbb_model.load_weights(PATH + 'bbb_model')
        
        no_samples = 20
        true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim* input_dim ])
        true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        mu_out_ = np.zeros([no_samples, int(x_test.shape[0] / (batch_size)), batch_size, class_num])
     #   acc_test1 = np.zeros([no_samples, int(x_test.shape[0] / (batch_size)),batch_size ])
        acc_test = np.zeros([no_samples, int(x_test.shape[0] / (batch_size))]) 
        for i in range(no_samples):
            test_no_steps = 0
            for step, (x, y) in enumerate(val_dataset):
                update_progress(step / int(x_test.shape[0] / (batch_size)) ) 
                true_x[test_no_steps, :, :] = x
                true_y[test_no_steps, :, :] = y
                if Random_noise:
                    noise = tf.random.normal(shape = [batch_size, input_dim* input_dim], mean = 0.0, stddev = gaussain_noise_std, dtype = x.dtype ) 
                    x = x +  noise
                mu_out   = test_on_batch(x, y)              
                mu_out_[i, test_no_steps,:,:] = mu_out           
                corr = tf.equal(tf.math.argmax(mu_out, axis=1),tf.math.argmax(y,axis=1))
                accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
                acc_test[i,test_no_steps] = accuracy.numpy()
            #    acc_test1[i,test_no_steps, :]=    tf.cast(corr, tf.float32) 
                if step % 100 == 0:
                    print("Total running accuracy so far: %.3f" % accuracy.numpy())             
                test_no_steps+=1      
        print('Sample variance on prediction : ', np.mean(np.var(mu_out_, axis=0)))      
        test_acc = np.mean(acc_test)          
        print('Test accuracy : ', test_acc)                  
     #   test_acc_std = np.mean(np.std(acc_test, axis=0)  )             
     #   print('STD Test Accuracy : ', test_acc_std )
        pf = open(PATH + test_path + 'uncertainty_info.pkl', 'wb')      
        pickle.dump([mu_out_, true_y,test_acc ], pf)                                                  
        pf.close()
        
        valid_size = x_test.shape[0]  
        pred_var = np.zeros(int(valid_size ))   
        true_var = np.zeros(int(valid_size )) 
        correct_classification = np.zeros(int(valid_size)) 
        misclassification_pred = np.zeros(int(valid_size )) 
        misclassification_true = np.zeros(int(valid_size )) 
        predicted_out = np.zeros(int(valid_size )) 
        true_out = np.zeros(int(valid_size )) 
        k=0   
        k1=0
        k2=0  
        correct_classification1 = 0
        misclassification_pred1 = 0
        misclassification_true1 = 0
      #  misclassification_pred1_ = 0
      #  correct_classification1_ = 0
     #   sample_var = np.std(acc_test1, axis=0) # shape=(int(valid_size /batch_size), batch_size)
        sample_var = np.var(mu_out_, axis=0)# shape=(int(valid_size /batch_size), batch_size,class_num )
        for i in range(int(valid_size /batch_size)):
            for j in range(batch_size):               
                predicted_out[k] = np.argmax( np.mean(mu_out_[:,i,j,:]), axis=0)
                true_out[k] = np.argmax(true_y[i,j,:])
                pred_var[k] =   np.mean(sample_var[i,j, :])                 
               # true_var[k] = sigma_[i,j, int(true_out[k]), int(true_out[k])]  
                if (predicted_out[k] == true_out[k]):
                    correct_classification[k1] = np.mean(sample_var[i,j, :])  
                    correct_classification1 +=  np.mean(sample_var[i,j, :])  
                  #  correct_classification1_ +=  np.square(sample_var[i,j])
                    k1=k1+1
                if (predicted_out[k] != true_out[k]):
                    misclassification_pred[k2] = np.mean(sample_var[i,j, :]) 
                    misclassification_pred1 +=  np.mean(sample_var[i,j, :])   
                  #  misclassification_pred1_ +=  np.square(sample_var[i,j])                    
                    k2=k2+1                 
                k=k+1  
                       
   #     print('Average Output Variance', np.mean(np.square(pred_var)))   
        correct_classification2 = correct_classification1/k1
        misclassification_pred2 = misclassification_pred1/k2
      #  correct_classification2_ = correct_classification1_/k1
      #  misclassification_pred2_ = misclassification_pred1_/k2
                     
        writer = pd.ExcelWriter(PATH + test_path + 'variance.xlsx', engine='xlsxwriter')
        df = pd.DataFrame(pred_var)   
        # Write your DataFrame to a file   
        df.to_excel(writer, "Sheet")  
        
        #df0 = pd.DataFrame(pred_var)   
        # Write your DataFrame to a file   
        #df0.to_excel(writer, "Sheet", startcol=4)    
        
        df1 = pd.DataFrame(predicted_out)
        df1.to_excel(writer, 'Sheet',  startcol=4)
        
        df2 = pd.DataFrame(true_out)
        df2.to_excel(writer, 'Sheet',  startcol=7)
          
       # df3 = pd.DataFrame(np.square(correct_classification))
       # df3.to_excel(writer, 'Sheet',  startcol=12)
        
        df4 = pd.DataFrame(correct_classification)
        df4.to_excel(writer, 'Sheet',  startcol=10)
        
       # df5 = pd.DataFrame(np.square(misclassification_pred))
       # df5.to_excel(writer, 'Sheet',  startcol=18)
        
        df6 = pd.DataFrame(misclassification_pred)
        df6.to_excel(writer, 'Sheet',  startcol=13)
        writer.save()
        pf = open(PATH + test_path + 'var_info.pkl', 'wb')                   
        pickle.dump([correct_classification, misclassification_true, pred_var], pf)                                                  
        pf.close()
        if Random_noise:
           snr_signal = np.zeros([int(x_test.shape[0] / (batch_size)) ,batch_size])
           for i in range(int(x_test.shape[0] / (batch_size))):
               for j in range(batch_size):
                   noise = tf.random.normal(shape = [input_dim, input_dim, 1], mean = 0.0, stddev = gaussain_noise_std, dtype = x.dtype ) 
                   snr_signal[i,j] = 10*np.log10( np.sum(np.square(true_x[i,j,:]))/np.sum( np.square(noise) ))         
           print('SNR', np.mean(snr_signal)) 
          
        writer.save()      
        textfile = open(PATH + test_path + 'Related_hyperparameters.txt','w')   
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No of units : ' +str(num_unit))
        textfile.write('\n Number of Classes : ' +str(class_num))
        textfile.write('\n No of epochs : ' +str(epochs))        
        textfile.write('\n Initial Learning rate : ' +str(lr)) 
        textfile.write('\n Ending Learning rate : ' +str(lr_end)) 
        textfile.write('\n batch size : ' +str(batch_size)) 
        textfile.write('\n KL term factor : ' +str(kl_factor))        
        textfile.write("\n---------------------------------")
        textfile.write("\n Test Accuracy : "+ str( test_acc)) 
        #textfile.write("\n Output Variance: "+ str(np.mean(np.square(pred_var)))) 
        textfile.write("\n Output Variance: "+ str(np.mean(pred_var))) 
        #textfile.write("\n test_acc_std: "+ str(test_acc_std)) 
        textfile.write("\n Correct Classification var: "+ str(correct_classification2)) 
        textfile.write("\n MisClassification var: "+ str(misclassification_pred2))
        #textfile.write("\n Correct Classification Variance: "+ str(correct_classification2_)) 
        #textfile.write("\n MisClassification Variance: "+ str(misclassification_pred2_))          
        textfile.write("\n---------------------------------")
        if Random_noise:
            textfile.write('\n Random Noise std: '+ str(gaussain_noise_std ))   
            textfile.write("\n SNR: "+ str(np.mean(snr_signal)))           
        textfile.write("\n---------------------------------")    
        textfile.close()
        
    elif (Testing):
        test_path = '/updated/test_results/'
        if Random_noise:
            test_path = '/updated/test_results_random_noise_{}/'.format(gaussain_noise_std)
            if not os.path.exists(PATH+ test_path ):
                os.makedirs(PATH+ test_path )
            #os.makedirs(PATH + test_path)
        else:
             if not os.path.exists(PATH+ test_path ):
                os.makedirs(PATH+ test_path )
           # os.makedirs(PATH + test_path)  
            
        
        bbb_model.load_weights(PATH + 'bbb_model') 
        no_samples = 20
        true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim* input_dim ])
        true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        mu_out_ = np.zeros([no_samples, int(x_test.shape[0] / (batch_size)), batch_size, class_num])
    #    acc_test1 = np.zeros([no_samples, int(x_test.shape[0] / (batch_size)),batch_size ])
        acc_test = np.zeros([no_samples, int(x_test.shape[0] / (batch_size))])     
        for i in range(no_samples):
            test_no_steps = 0
            for step, (x, y) in enumerate(val_dataset):
                update_progress(step / int(x_test.shape[0] / (batch_size)) ) 
                true_x[test_no_steps, :, :] = x
                true_y[test_no_steps, :, :] = y
                if Random_noise:
                    noise = tf.random.normal(shape = [batch_size, input_dim* input_dim], mean = 0.0, stddev = gaussain_noise_std, dtype = x.dtype ) 
                    x = x +  noise
                mu_out   = test_on_batch(x, y)              
                mu_out_[i, test_no_steps,:,:] = mu_out           
                corr = tf.equal(tf.math.argmax(mu_out, axis=1),tf.math.argmax(y,axis=1))
                accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
                acc_test[i,test_no_steps] = accuracy.numpy()
          #      acc_test1[i,test_no_steps, :]=    tf.cast(corr, tf.float32) 
                if step % 100 == 0:
                    print("Total running accuracy so far: %.3f" % accuracy.numpy())             
                test_no_steps+=1      
        print('Sample variance on prediction : ', np.mean(np.var(mu_out_, axis=0)))      
        test_acc = np.mean(acc_test)          
        print('Test accuracy : ', test_acc)                  
       # test_acc_std = np.mean(np.std(acc_test, axis=0)  )             
      #  print('STD Test Accuracy : ', test_acc_std )
        pf = open(PATH + test_path + 'uncertainty_info.pkl', 'wb')      
        pickle.dump([mu_out_, true_y,test_acc ], pf)                                                  
        pf.close()
        
        valid_size = x_test.shape[0]  
        pred_var = np.zeros(int(valid_size ))   
        true_var = np.zeros(int(valid_size )) 
        correct_classification = np.zeros(int(valid_size)) 
        misclassification_pred = np.zeros(int(valid_size )) 
        misclassification_true = np.zeros(int(valid_size )) 
        predicted_out = np.zeros(int(valid_size )) 
        true_out = np.zeros(int(valid_size )) 
        k=0   
        k1=0
        k2=0  
        correct_classification1 = 0
        misclassification_pred1 = 0
        misclassification_true1 = 0
     #   misclassification_pred1_ = 0
      #  correct_classification1_ = 0
       # sample_var = np.std(acc_test1, axis=0) # shape=(int(valid_size /batch_size), batch_size)
        sample_var = np.var(mu_out_, axis=0)# shape=(int(valid_size /batch_size), batch_size,class_num )
        for i in range(int(valid_size /batch_size)):
            for j in range(batch_size):               
                predicted_out[k] = np.argmax( np.mean(mu_out_[:,i,j,:]), axis=0)
                true_out[k] = np.argmax(true_y[i,j,:])
                pred_var[k] =   np.mean(sample_var[i,j, :])                 
               # true_var[k] = sigma_[i,j, int(true_out[k]), int(true_out[k])]  
                if (predicted_out[k] == true_out[k]):
                    correct_classification[k1] = np.mean(sample_var[i,j, :]) 
                    correct_classification1 +=  np.mean(sample_var[i,j, :])  
               #     correct_classification1_ +=  np.square(sample_var[i,j])
                    k1=k1+1
                if (predicted_out[k] != true_out[k]):
                    misclassification_pred[k2] = np.mean(sample_var[i,j, :])  
                    misclassification_pred1 +=  np.mean(sample_var[i,j, :])   
              #      misclassification_pred1_ +=  np.square(sample_var[i,j])                    
                    k2=k2+1                 
                k=k+1         
      #  print('Average Output Variance', np.mean(np.square(pred_var)))   
        correct_classification2 = correct_classification1/k1
        misclassification_pred2 = misclassification_pred1/k2
     #   correct_classification2_ = correct_classification1_/k1
      #  misclassification_pred2_ = misclassification_pred1_/k2
                     
        writer = pd.ExcelWriter(PATH + test_path + 'variance.xlsx', engine='xlsxwriter')
        df = pd.DataFrame(pred_var)   
        # Write your DataFrame to a file   
        df.to_excel(writer, "Sheet")  
        
       #df0 = pd.DataFrame(pred_var)   
        # Write your DataFrame to a file   
       # df0.to_excel(writer, "Sheet", startcol=4)    
        
        df1 = pd.DataFrame(predicted_out)
        df1.to_excel(writer, 'Sheet',  startcol=4)
        
        df2 = pd.DataFrame(true_out)
        df2.to_excel(writer, 'Sheet',  startcol=7)
          
       # df3 = pd.DataFrame(np.square(correct_classification))
       # df3.to_excel(writer, 'Sheet',  startcol=12)
        
        df4 = pd.DataFrame(correct_classification)
        df4.to_excel(writer, 'Sheet',  startcol=10)
        
      #  df5 = pd.DataFrame(np.square(misclassification_pred))
       # df5.to_excel(writer, 'Sheet',  startcol=18)
        
        df6 = pd.DataFrame(misclassification_pred)
        df6.to_excel(writer, 'Sheet',  startcol=13)
        writer.save()
        pf = open(PATH + test_path + 'var_info.pkl', 'wb')                   
        pickle.dump([correct_classification, misclassification_true, pred_var], pf)                                                  
        pf.close()
        if Random_noise:
           snr_signal = np.zeros([int(x_test.shape[0] / (batch_size)) ,batch_size])
           for i in range(int(x_test.shape[0] / (batch_size))):
               for j in range(batch_size):
                   noise = tf.random.normal(shape = [input_dim, input_dim, 1], mean = 0.0, stddev = gaussain_noise_std, dtype = x.dtype ) 
                   snr_signal[i,j] = 10*np.log10( np.sum(np.square(true_x[i,j,:]))/np.sum( np.square(noise) ))         
           print('SNR', np.mean(snr_signal)) 
          
        writer.save()      
        textfile = open(PATH + test_path + 'Related_hyperparameters.txt','w')   
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No of units : ' +str(num_unit))
        textfile.write('\n Number of Classes : ' +str(class_num))
        textfile.write('\n No of epochs : ' +str(epochs))        
        textfile.write('\n Initial Learning rate : ' +str(lr)) 
        textfile.write('\n Ending Learning rate : ' +str(lr_end)) 
        textfile.write('\n batch size : ' +str(batch_size)) 
        textfile.write('\n KL term factor : ' +str(kl_factor))        
        textfile.write("\n---------------------------------")
        textfile.write("\n Test Accuracy : "+ str( test_acc)) 
       # textfile.write("\n Output Variance: "+ str(np.mean(np.square(pred_var)))) 
        textfile.write("\n Output variance: "+ str(np.mean(pred_var))) 
       # textfile.write("\n test_acc_std: "+ str(test_acc_std)) 
        textfile.write("\n Correct Classification std: "+ str(correct_classification2)) 
        textfile.write("\n MisClassification std: "+ str(misclassification_pred2))
       # textfile.write("\n Correct Classification Variance: "+ str(correct_classification2_)) 
       # textfile.write("\n MisClassification Variance: "+ str(misclassification_pred2_))          
        textfile.write("\n---------------------------------")
        if Random_noise:
            textfile.write('\n Random Noise std: '+ str(gaussain_noise_std ))   
            textfile.write("\n SNR: "+ str(np.mean(snr_signal)))           
        textfile.write("\n---------------------------------")    
        textfile.close()
    if(corrupted_images):
                #-------------------------Testing with corrupted MNIST----------------------------- 
       # test_path1 = '/updated/'        
        if kind=="Glass":
            print('working with glass')
            test_path = 'updated/Corrupted_Images/Glass_Blur/'
            x_test = np.load('./mnist_c/glass_blur/test_images.npy')
            y_test = np.load('./mnist_c/glass_blur/test_labels.npy')
        elif kind =="Motion":
            test_path = 'updated/Corrupted_Images/Motion_Blur/'
            x_test = np.load('./mnist_c/motion_blur/test_images.npy')
            y_test = np.load('./mnist_c/motion_blur/test_labels.npy')
        elif kind =="Zigzag":
            test_path = 'updated/Corrupted_Images/Zigzag/'  
            x_test = np.load('./mnist_c/zigzag/test_images.npy')
            y_test = np.load('./mnist_c/zigzag/test_labels.npy')
        elif kind =="Dotted":
            test_path = 'updated/Corrupted_Images/Dotted_Line/'
            x_test = np.load('./mnist_c/dotted_line/test_images.npy')
            y_test = np.load('./mnist_c/dotted_line/test_labels.npy')
        if kind=="Scale":
            test_path = 'updated/Corrupted_Images/Scale/'
            x_test = np.load('./mnist_c/scale/test_images.npy')
            y_test = np.load('./mnist_c/scale/test_labels.npy')
        elif kind =="Spatter":
            test_path = 'updated/Corrupted_Images/Spatter/'
            x_test = np.load('./mnist_c/spatter/test_images.npy')
            y_test = np.load('./mnist_c/spatter/test_labels.npy')
        elif kind =="Brightness":
            test_path = 'updated/Corrupted_Images/Brightness/'
            x_test = np.load('./mnist_c/brightness/test_images.npy')
            y_test = np.load('./mnist_c/brightness/test_labels.npy')
        elif kind =="Shear":
            test_path = 'updated/Corrupted_Images/Shear/'
            x_test = np.load('./mnist_c/shear/test_images.npy')
            y_test = np.load('./mnist_c/shear/test_labels.npy')    

        if kind=="Identity":
            test_path = 'updated/Corrupted_Images/Identity/'
            x_test = np.load('./mnist_c/identity/test_images.npy')
            y_test = np.load('./mnist_c/identity/test_labels.npy')
        elif kind =="Shot":
            test_path = 'updated/Corrupted_Images/Shot/'
            x_test = np.load('./mnist_c/shot_noise/test_images.npy')
            y_test = np.load('./mnist_c/shot_noise/test_labels.npy')
        elif kind =="Stripe":
            test_path = 'updated/Corrupted_Images/Stripe/'
            x_test = np.load('./mnist_c/stripe/test_images.npy')
            y_test = np.load('./mnist_c/stripe/test_labels.npy')
        elif kind =="Fog":
            test_path = 'updated/Corrupted_Images/Fog/'
            x_test = np.load('./mnist_c/fog/test_images.npy')
            y_test = np.load('./mnist_c/fog/test_labels.npy')  
        if kind=="Translate":
            test_path = 'updated/Corrupted_Images/Translate/'
            x_test = np.load('./mnist_c/translate/test_images.npy')
            y_test = np.load('./mnist_c/translate/test_labels.npy')
        elif kind =="Rotate":
            test_path = 'updated/Corrupted_Images/Rotate/'
            x_test = np.load('./mnist_c/rotate/test_images.npy')
            y_test = np.load('./mnist_c/rotate/test_labels.npy')
        elif kind =="Canny":
            test_path = 'updated/Corrupted_Images/Canny_edge/'
            x_test = np.load('./mnist_c/canny_edges/test_images.npy')
            y_test = np.load('./mnist_c/canny_edges/test_labels.npy')
        elif kind =="Impulse":
            test_path = 'updated/Corrupted_Images/Impulse_noise/'
            x_test = np.load('./mnist_c/impulse_noise/test_images.npy')
            y_test = np.load('./mnist_c/impulse_noise/test_labels.npy')
            
        if not os.path.exists(PATH+ test_path ):
              os.makedirs(PATH+ test_path )
      #  os.makedirs(PATH + test_path1 + test_path)
        #test_path = 'test_results_random_noise_{}/'.format(gaussain_noise_std)
        bbb_model.load_weights(PATH + 'bbb_model') 
               
        x_test =  x_test / 255.0
        x_test = tf.reshape(tf.cast(x_test,tf.float32), [-1, 28*28]) 
        one_hot_y_test = tf.one_hot(y_test.astype(np.float32), depth=class_num)
        #y_test = y_test.astype(np.int32)
        val_dataset1 = tf.data.Dataset.from_tensor_slices((x_test, one_hot_y_test)).batch(batch_size)
        valid_size = x_test.shape[0]
        no_samples = 20

        true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim* input_dim])        
        true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        mu_out_ = np.zeros([no_samples, int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        #acc_test1 = np.zeros([no_samples, int(x_test.shape[0] / (batch_size)),batch_size ])
        acc_test = np.zeros(int(x_test.shape[0] / (batch_size)))
        for i in range(no_samples):
            test_no_steps = 0 
            for step, (x, y) in enumerate(val_dataset1):
                #y = tf.one_hot(y.astype(np.float32), depth=class_num)
                update_progress(step / int(x_test.shape[0] / (batch_size)) ) 
                true_x[test_no_steps, :, :] = x
                true_y[test_no_steps, :, :] = y
                
                mu_out   = test_on_batch(x, y)              
                mu_out_[i, test_no_steps,:,:] = mu_out           
                corr = tf.equal(tf.math.argmax(mu_out, axis=1),tf.math.argmax(y,axis=1))
                accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
                acc_test[test_no_steps] = accuracy.numpy()
                #acc_test1[i,test_no_steps, :]=    tf.cast(corr, tf.float32) 
                if step % 100 == 0:
                    print("Total running accuracy so far: %.3f" % accuracy.numpy())             
                test_no_steps+=1
                              
        print('Sample variance on prediction : ', np.mean(np.var(mu_out_, axis=0)))         
        test_acc = np.mean(acc_test)          
        print('Test accuracy : ', test_acc)                  
      #  test_acc_std = np.mean(np.std(acc_test, axis=0)  )             
       # print('STD Test Accuracy : ', test_acc_std )
        pf = open(PATH + test_path + 'uncertainty_info.pkl', 'wb')      
        pickle.dump([mu_out_, true_y,test_acc ], pf)                                                  
        pf.close()
        
        valid_size = x_test.shape[0]  
        pred_var = np.zeros(int(valid_size ))   
        true_var = np.zeros(int(valid_size )) 
        correct_classification = np.zeros(int(valid_size)) 
        misclassification_pred = np.zeros(int(valid_size )) 
        misclassification_true = np.zeros(int(valid_size )) 
        predicted_out = np.zeros(int(valid_size )) 
        true_out = np.zeros(int(valid_size )) 
        k=0   
        k1=0
        k2=0  
        correct_classification1 = 0
        misclassification_pred1 = 0
        misclassification_true1 = 0
       # misclassification_pred1_ = 0
       # correct_classification1_ = 0
     #   sample_var = np.std(acc_test1, axis=0) # shape=(int(valid_size /batch_size), batch_size)
        sample_var = np.var(mu_out_, axis=0)# shape=(int(valid_size /batch_size), batch_size,class_num )
        for i in range(int(valid_size /batch_size)):
            for j in range(batch_size):               
                predicted_out[k] = np.argmax( np.mean(mu_out_[:,i,j,:]), axis=0)
                true_out[k] = np.argmax(true_y[i,j,:])
                pred_var[k] =   np.mean(sample_var[i,j, :])                 
               # true_var[k] = sigma_[i,j, int(true_out[k]), int(true_out[k])]  
                if (predicted_out[k] == true_out[k]):
                    correct_classification[k1] = np.mean(sample_var[i,j, :]) 
                    correct_classification1 +=  np.mean(sample_var[i,j, :]) 
                  #  correct_classification1_ +=  np.square(sample_var[i,j])
                    k1=k1+1
                if (predicted_out[k] != true_out[k]):
                    misclassification_pred[k2] = np.mean(sample_var[i,j, :]) 
                    misclassification_pred1 +=  np.mean(sample_var[i,j, :])  
                  #  misclassification_pred1_ +=  np.square(sample_var[i,j])                    
                    k2=k2+1                 
                k=k+1         
      #  print('Average Output Variance', np.mean(np.square(pred_var)))   
        correct_classification2 = correct_classification1/k1
        misclassification_pred2 = misclassification_pred1/k2
       # correct_classification2_ = correct_classification1_/k1
       # misclassification_pred2_ = misclassification_pred1_/k2
                     
        writer = pd.ExcelWriter(PATH + test_path + 'variance.xlsx', engine='xlsxwriter')
        df = pd.DataFrame(pred_var)  
        # Write your DataFrame to a file   
        df.to_excel(writer, "Sheet")  
        
        #df0 = pd.DataFrame(pred_var)   
        # Write your DataFrame to a file   
        #df0.to_excel(writer, "Sheet", startcol=4)    
        
        df1 = pd.DataFrame(predicted_out)
        df1.to_excel(writer, 'Sheet',  startcol=4)
        
        df2 = pd.DataFrame(true_out)
        df2.to_excel(writer, 'Sheet',  startcol=7)
          
       # df3 = pd.DataFrame(np.square(correct_classification))
      #  df3.to_excel(writer, 'Sheet',  startcol=12)
        
        df4 = pd.DataFrame(correct_classification)
        df4.to_excel(writer, 'Sheet',  startcol=10)
        
      #  df5 = pd.DataFrame(np.square(misclassification_pred))
       # df5.to_excel(writer, 'Sheet',  startcol=18)
        
        df6 = pd.DataFrame(misclassification_pred)
        df6.to_excel(writer, 'Sheet',  startcol=13)
        writer.save()
        pf = open(PATH + test_path + 'var_info.pkl', 'wb')                   
        pickle.dump([correct_classification, misclassification_true, pred_var], pf)                                                  
        pf.close()
 
        textfile = open(PATH + test_path + 'Related_hyperparameters.txt','w')   
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No of units : ' +str(num_unit))
        textfile.write('\n Number of Classes : ' +str(class_num))
        textfile.write('\n No of epochs : ' +str(epochs))        
        textfile.write('\n Initial Learning rate : ' +str(lr)) 
        textfile.write('\n Ending Learning rate : ' +str(lr_end)) 
        textfile.write('\n batch size : ' +str(batch_size)) 
        textfile.write('\n KL term factor : ' +str(kl_factor))        
        textfile.write("\n---------------------------------")
        textfile.write("\n Test Accuracy : "+ str( test_acc)) 
   #     textfile.write("\n Output Variance: "+ str(np.mean(np.square(pred_var)))) 
        textfile.write("\n Output Variance: "+ str(np.mean(pred_var))) 
      #  textfile.write("\n test_acc_std: "+ str(test_acc_std)) 
        textfile.write("\n Correct Classification std: "+ str(correct_classification2)) 
        textfile.write("\n MisClassification std: "+ str(misclassification_pred2))
      #  textfile.write("\n Correct Classification Variance: "+ str(correct_classification2_)) 
      #  textfile.write("\n MisClassification Variance: "+ str(misclassification_pred2_))                  
        textfile.write("\n---------------------------------")  
        textfile.close()
    
    elif(Adversarial_noise):
        if Targeted:
            test_path = '/updated/test_results_targeted_adversarial_noise_{}/'.format(epsilon)  
            if not os.path.exists(PATH+ test_path ):
                os.makedirs(PATH+ test_path )
         #   os.makedirs(PATH + test_path)          
        else:
            test_path = '/updated/test_results_non_targeted_adversarial_noise_{}/'.format(epsilon) 
            if not os.path.exists(PATH+ test_path ):
                os.makedirs(PATH+ test_path )
        #    os.makedirs(PATH + test_path)             
        bbb_model.load_weights(PATH + 'bbb_model')       
               
        no_samples = 20
        true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim* input_dim])
        adv_perturbations = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim* input_dim])
        true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        mu_out_ = np.zeros([no_samples, int(x_test.shape[0] / (batch_size)), batch_size, class_num])
      #  acc_test1 = np.zeros([no_samples, int(x_test.shape[0] / (batch_size)),batch_size ])
        acc_test = np.zeros(int(x_test.shape[0]/(batch_size)))
        for i in range(no_samples):
            test_no_steps = 0 
            for step, (x, y) in enumerate(val_dataset):
                update_progress(step / int(x_test.shape[0] / (batch_size)) ) 
                true_x[test_no_steps, :, :] = x
                true_y[test_no_steps, :, :] = y
                
                if Targeted:
                    y_true_batch = np.zeros_like(y)
                    y_true_batch[:, adversary_target_cls] = 1.0            
                    adv_perturbations[test_no_steps, :, :] = create_adversarial_pattern(x, y_true_batch)
                else:
                    adv_perturbations[test_no_steps, :, :] = create_adversarial_pattern(x, y)
                adv_x = x + epsilon*adv_perturbations[test_no_steps, :, :] 
                adv_x = tf.clip_by_value(adv_x, 0.0, 1.0) 
                
                mu_out   = test_on_batch(adv_x, y)           
                mu_out_[i, test_no_steps,:,:] = mu_out            
                corr = tf.equal(tf.math.argmax(mu_out, axis=1),tf.math.argmax(y,axis=1))
                accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
                acc_test[test_no_steps]=accuracy.numpy()
               # acc_test1[i,test_no_steps, :]=    tf.cast(corr, tf.float32) 
                if step % 10 == 0:
                    print("Total running accuracy so far: %.3f" % accuracy.numpy())             
                test_no_steps+=1 
                         
        print('Sample variance on prediction : ', np.mean(np.var(mu_out_, axis=0))) 
        test_acc = np.mean(acc_test)          
        print('Test accuracy : ', test_acc)                  
      #  test_acc_std = np.mean(np.std(acc_test, axis=0)  )             
     #   print('STD Test Accuracy : ', test_acc_std )
        pf = open(PATH + test_path + 'uncertainty_info.pkl', 'wb')      
        pickle.dump([mu_out_, true_y,test_acc, adv_perturbations ], pf)                                                  
        pf.close()          
        
        snr_signal = np.zeros([int(x_test.shape[0] /batch_size) ,batch_size])
        for i in range(int(x_test.shape[0] /batch_size)):
            for j in range(batch_size):    
                snr_signal[i,j] = 10*np.log10( np.sum(np.square(true_x[i,j,:]))/np.sum( np.square(epsilon*adv_perturbations[i, j, :]  ) ))       
        print('SNR', np.mean(snr_signal))
        
         
        valid_size = x_test.shape[0]  
        pred_var = np.zeros(int(valid_size ))   
        true_var = np.zeros(int(valid_size )) 
        correct_classification = np.zeros(int(valid_size)) 
        misclassification_pred = np.zeros(int(valid_size )) 
        misclassification_true = np.zeros(int(valid_size )) 
        predicted_out = np.zeros(int(valid_size )) 
        true_out = np.zeros(int(valid_size )) 
        k=0   
        k1=0
        k2=0  
        correct_classification1 = 0
        misclassification_pred1 = 0
        misclassification_true1 = 0
      #  misclassification_pred1_ = 0
      #  correct_classification1_ = 0
       # sample_var = np.std(acc_test1, axis=0) # shape=(int(valid_size /batch_size), batch_size)
        sample_var = np.var(mu_out_, axis=0)# shape=(int(valid_size /batch_size), batch_size,class_num )
        for i in range(int(valid_size /batch_size)):
            for j in range(batch_size):               
                predicted_out[k] = np.argmax( np.mean(mu_out_[:,i,j,:]), axis=0)
                true_out[k] = np.argmax(true_y[i,j,:])
                pred_var[k] =   np.mean(sample_var[i,j, :])                 
               # true_var[k] = sigma_[i,j, int(true_out[k]), int(true_out[k])]  
                if (predicted_out[k] == true_out[k]):
                    correct_classification[k1] = np.mean(sample_var[i,j, :])  
                    correct_classification1 +=  np.mean(sample_var[i,j, :]) 
                  #  correct_classification1_ +=  np.square(sample_var[i,j])
                    k1=k1+1
                if (predicted_out[k] != true_out[k]):
                    misclassification_pred[k2] = np.mean(sample_var[i,j, :])  
                    misclassification_pred1 +=  np.mean(sample_var[i,j, :]) 
                    #misclassification_pred1_ +=  np.square(sample_var[i,j])                    
                    k2=k2+1                 
                k=k+1         
        #print('Average Output Variance', np.mean(np.square(pred_var)))   
        correct_classification2 = correct_classification1/k1
        misclassification_pred2 = misclassification_pred1/k2
       # correct_classification2_ = correct_classification1_/k1
       # misclassification_pred2_ = misclassification_pred1_/k2
                     
        writer = pd.ExcelWriter(PATH + test_path + 'variance.xlsx', engine='xlsxwriter')
        df = pd.DataFrame(pred_var)  
        # Write your DataFrame to a file   
        df.to_excel(writer, "Sheet")  
        
       # df0 = pd.DataFrame(pred_var)   
        # Write your DataFrame to a file   
        #df0.to_excel(writer, "Sheet", startcol=4)    
        
        df1 = pd.DataFrame(predicted_out)
        df1.to_excel(writer, 'Sheet',  startcol=4)
        
        df2 = pd.DataFrame(true_out)
        df2.to_excel(writer, 'Sheet',  startcol=7)
          
        #df3 = pd.DataFrame(np.square(correct_classification))
        #df3.to_excel(writer, 'Sheet',  startcol=12)
        
        df4 = pd.DataFrame(correct_classification)
        df4.to_excel(writer, 'Sheet',  startcol=10)
        
        #df5 = pd.DataFrame(np.square(misclassification_pred))
       # df5.to_excel(writer, 'Sheet',  startcol=18)
        
        df6 = pd.DataFrame(misclassification_pred)
        df6.to_excel(writer, 'Sheet',  startcol=13)
        writer.save()
        pf = open(PATH + test_path + 'var_info.pkl', 'wb')                   
        pickle.dump([correct_classification, misclassification_true, pred_var], pf)                                                  
        pf.close() 
        
        textfile = open(PATH + test_path + 'Related_hyperparameters.txt','w')   
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No of units : ' +str(num_unit))
        textfile.write('\n Number of Classes : ' +str(class_num))
        textfile.write('\n No of epochs : ' +str(epochs))
        textfile.write('\n Initial Learning rate : ' +str(lr)) 
        textfile.write('\n Ending Learning rate : ' +str(lr_end))  
        textfile.write('\n batch size : ' +str(batch_size)) 
        textfile.write('\n KL term factor : ' +str(kl_factor))      
        textfile.write("\n---------------------------------")
        textfile.write("\n Averaged Test Accuracy : "+ str( test_acc))  
       # textfile.write("\n Output Variance: "+ str(np.mean(np.square(pred_var)))) 
        textfile.write("\n Output Variance: "+ str(np.mean(pred_var))) 
      #  textfile.write("\n test_acc_std: "+ str(test_acc_std)) 
        textfile.write("\n Correct Classification std: "+ str(correct_classification2)) 
        textfile.write("\n MisClassification std: "+ str(misclassification_pred2))
       # textfile.write("\n Correct Classification Variance: "+ str(correct_classification2_)) 
       # textfile.write("\n MisClassification Variance: "+ str(misclassification_pred2_))
        textfile.write("\n---------------------------------")
        if Adversarial_noise:
            if Targeted:
                textfile.write('\n Adversarial attack: TARGETED')
                textfile.write('\n The targeted attack class: ' + str(adversary_target_cls))                   
            else:      
                textfile.write('\n Adversarial attack: Non-TARGETED')
            textfile.write('\n Adversarial Noise epsilon: '+ str(epsilon ))    
            textfile.write("\n SNR: "+ str(np.mean(snr_signal)))               
        textfile.write("\n---------------------------------")    
        textfile.close()   
if __name__ == '__main__':
    main_function(Testing=True)
#    main_function( Testing=True, Random_noise=True, gaussain_noise_std=0.001)
#    main_function( Testing=True, Random_noise=True, gaussain_noise_std=0.01)
#    main_function(Testing=True, Random_noise=True, gaussain_noise_std=0.1)
#    main_function(Testing=True, Random_noise=True, gaussain_noise_std=0.2)
#    main_function(Testing=True, Random_noise=True, gaussain_noise_std=0.3)
#    main_function(Testing=True, Random_noise=True, gaussain_noise_std=0.4)
#    main_function(Testing=True, Random_noise=True, gaussain_noise_std=0.5)

#    main_function(Adversarial_noise=True, Targeted=False,epsilon = 0.001)
#    main_function(Adversarial_noise=True, Targeted=False,epsilon = 0.005)
#    main_function(Adversarial_noise=True, Targeted=False,epsilon = 0.01)
#    main_function(Adversarial_noise=True, Targeted=False,epsilon = 0.05)
#    main_function(Adversarial_noise=True, Targeted=False,epsilon = 0.1)
#    main_function(Adversarial_noise=True, Targeted=False,epsilon = 0.2)
#    
#    main_function(Adversarial_noise=True, Targeted=True,epsilon = 0.001)
#    main_function(Adversarial_noise=True, Targeted=True,epsilon = 0.005)
#    main_function(Adversarial_noise=True, Targeted=True,epsilon = 0.01)
#    main_function(Adversarial_noise=True, Targeted=True,epsilon = 0.05)
#    main_function(Adversarial_noise=True, Targeted=True,epsilon = 0.1)
#    main_function(Adversarial_noise=True, Targeted=True,epsilon = 0.2)
    
    main_function(corrupted_images=True,kind="Glass")
    main_function(corrupted_images=True,kind="Motion")
    main_function(corrupted_images=True,kind="Zigzag")
    main_function(corrupted_images=True,kind="Dotted")
    main_function(corrupted_images=True,kind="Scale")
    main_function(corrupted_images=True,kind="Spatter")
    main_function(corrupted_images=True,kind="Brightness")
    main_function(corrupted_images=True,kind="Shear")
    main_function(corrupted_images=True,kind="Identity")
    main_function(corrupted_images=True,kind="Shot")
    main_function(corrupted_images=True,kind="Stripe")
    main_function(corrupted_images=True,kind="Fog")
    main_function(corrupted_images=True,kind="Translate")
    main_function(corrupted_images=True,kind="Rotate")
    main_function(corrupted_images=True,kind="Canny")
    main_function(corrupted_images=True,kind="Impulse")


