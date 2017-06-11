#! /usr/bin/python
import tensorflow as tf
import matplotlib.pyplot as plt
import gan_model
import database

# Bench parameters
gd_training_epochs = 5
gi_training_epochs = 2
db_type=1              # 0 for images, 1 for auto generated shapes
# Used on only when db_type=1
db_size=8
# Used only when db_type=0
db_path=''
bench_name='rectangles_db'
seed = 32
img_size=64
z_length=100
batch_size=500
learning_rate_G = 0.0002
learning_rate_D = 0.0002
D_train_epochs=1
G_train_epochs=2

# Result's presentation parameters
display_step = 1
save_step = max(1,int(gd_training_epochs*0.1))

# Inits
gan0=gan_model.gan(batch_size,img_size,z_length,seed)
gan0.Build_Model(learning_rate_G,learning_rate_D)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sav_GD=tf.train.Saver()
dat=database.Real_DB(db_type,batch_size,db_pt=db_path,img_size=img_size,db_size=db_size)

# Train
counter=gan0.Restore_Checkpoint(sess,sav_GD,checkpoint_dir=bench_name + '/checkpoints/GD')
costs=gan0.Train_GD(sess,dat,bench_name,gd_training_epochs,D_train_epochs,G_train_epochs,display_step=display_step,save_step=save_step,saving_obj=sav_GD,from_epoch=counter)

# Show Convergence Results
plt.figure(1)
plt.plot(costs[:,0],label="Cost D")
plt.plot(costs[:,1],label="Cost G")
plt.legend()
plt.savefig('GD_Training.png')
plt.show()
sess.close()