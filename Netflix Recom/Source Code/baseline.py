import time
import numpy as np

class Baseline:
  def __init__(self):
    self.bu = None
    self.bi = None
  
  def get_saved_params(self,iteration,base_path="/content/gdrive/MyDrive/baseline/"):
    '''
    '''
    bu_path=base_path+'bu_'+str(iteration)+'.npy'
    bi_path=base_path+'bi_'+str(iteration)+'.npy'

    def load_np_array(file_path):
      '''
      '''
      with open(file_path, 'rb') as f:
        x = np.load(f)

      return x
    
    self.bu=load_np_array(bu_path)
    self.bi=load_np_array(bi_path)

  def train(self,train_df,mapper=None,n_epochs=20,lr=0.007,lambda_reg=0.002):

    user_num = len(set(train_df['id_profile']))
    item_num = len(set(train_df['id_asset']))

    mu = train_df['interest'].sum()/(train_df.shape[0])

    self.bu = np.zeros(user_num, np.double)
    self.bi = np.zeros(item_num, np.double)

    for epoch in range(n_epochs):
      print(" Started epoch {}".format(epoch))

      t=time.process_time()
      e1=0
      for u,i,r in train_df.values:

        
        u_index=mapper.profile_dict[u]
        i_index=mapper.asset_dict[i]

        bui = mu + self.bu[u_index] + self.bi[i_index]

        self.bu[u_index] += lr * (2*(r-bui) - 2*lambda_reg * self.bu[u_index])
        self.bi[i_index] += lr * (2*(r-bui) - 2*lambda_reg * self.bi[i_index])
        
    
      e=time.process_time()-t
      print("\tFinished epoch:{} | Took:{:.2f}".format(epoch,e))
      with open('/content/gdrive/MyDrive/baseline/bu_'+str(epoch)+'.npy', 'wb') as f:
            np.save(f, self.bu)
      with open('/content/gdrive/MyDrive/baseline/bi_'+str(epoch)+'.npy', 'wb') as f:
            np.save(f, self.bi)
  
  def predict(self,user,asset,mu,mapper=None):
    '''
    '''
    u_index=mapper.profile_dict[user]
    i_index=mapper.asset_dict[asset]

    bui = mu + self.bu[u_index] + self.bi[i_index]
    
    return bui

  def get_rmse(self,df,mapper=None,mu=None):
    '''
    '''
    rmse=0
    x=0
    for u,i,r in df.values:
      if x%1000000==0:
        print(" Treated {:.2f} Millions".format(x/1000000))
      try:
        u_index=mapper.profile_dict[u]
        i_index=mapper.asset_dict[i]
        x+=1
      except:
        continue

      bui = mu + self.bu[u_index] + self.bi[i_index]

      rmse+=(bui-r)**2
    
    return rmse/df.shape[0]
