import time
import numpy as np

class SVDPP:
  def __init__(self):
    self.x=3

  def get_saved_params(self,iteration,base_path="/content/gdrive/MyDrive/svdpp/"):
    '''
    '''
    bu_path=base_path+'bu_'+str(iteration)+'.npy'
    bi_path=base_path+'bi_'+str(iteration)+'.npy'
    p_path=base_path+'p_'+str(iteration)+'.npy'
    q_path=base_path+'q_'+str(iteration)+'.npy'
    y_path=base_path+'y_'+str(iteration)+'.npy'

    def load_np_array(file_path):
      '''
      '''
      with open(file_path, 'rb') as f:
        x = np.load(f)

      return x
    
    self.bu=load_np_array(bu_path)
    self.bi=load_np_array(bi_path)
    self.p=load_np_array(p_path)
    self.q=load_np_array(q_path)
    self.y=load_np_array(y_path)

  def predict(self,user,asset,mu,Nu,mapper=None):
    '''
    '''
    #Nu=context.loc[user]['id_asset'].tolist()
    Nu_index = mapper.get_many(Nu,'asset')
        
    I_Nu = len(Nu_index)
    sqrt_N_u = np.sqrt(I_Nu)
    y_u = np.sum(self.y[Nu_index], axis=0)
    u_impl_prf = y_u / sqrt_N_u

    u_index=mapper.profile_dict[user]
    i_index=mapper.asset_dict[asset]

    rp = mu + self.bu[u_index] + self.bi[i_index] + np.dot(self.q[i_index], self.p[u_index] + u_impl_prf) 
      
    return rp

  def get_rmse(self,df,mapper=None,mu=None):
    '''
    '''
    rmse=0
    x=0

    df_indexed=df.set_index('id_profile',inplace=False)
    df_indexed.sort_index(inplace=True)
    t=time.process_time()
    for index,row in df_indexed.iterrows():
      if x%100000==0:
        e=time.process_time()-t
        print('Batch: {:.2f} Took: {:.2f}'.format(x/100000,e))
        t=time.process_time()
      x+=1

      u=index 
      i=row['id_asset']
      r=row['interest']

      u_index=mapper.profile_dict[u]
      i_index=mapper.asset_dict[i]

      Nu=df_indexed.loc[u]['id_asset'].tolist()
      Nu_index = mapper.get_many(Nu,'asset')
        
      I_Nu = len(Nu_index)
      sqrt_N_u = np.sqrt(I_Nu)
      y_u = np.sum(self.y[Nu_index], axis=0)
      u_impl_prf = y_u / sqrt_N_u
      
      rp = mu + self.bu[u_index] + self.bi[i_index] + np.dot(self.q[i_index], self.p[u_index] + u_impl_prf) 

      rmse+=(rp-r)**2
    
    return rmse/df.shape[0]
      

  def train(self,train_df,mapper=None,n_epochs=20,n_factors=20,lr=0.007,reg=0.002,is_trained=False,iteration=-1):
    m=mapper

    user_num = len(set(train_df['id_profile']))
    item_num = len(set(train_df['id_asset']))
    
    
    mu = train_df['interest'].sum()/(train_df.shape[0])

    np.random.RandomState(123)

    if not is_trained:
      print('initializing params randomly...')
      bu = np.zeros(user_num, np.double)
      bi = np.zeros(item_num, np.double)
      p = np.random.RandomState(123).normal(0, 0.1,(user_num, n_factors))
      q = np.random.RandomState(123).normal(0, 0.1,(item_num, n_factors))
      y = np.random.RandomState(123).normal(0, 0.1,(item_num, n_factors))
    else:
      print('Getting saved params of iteration: ', iteration)
      self.get_saved_params(iteration=iteration)
      bu,bi,p,q,y=self.bu,self.bi,self.p,self.q,self.y

    print('Indexing and sorting the dataframe...')
    indexed_df=train_df.set_index('id_profile')
    indexed_df.sort_index(inplace=True)

    for current_epoch in range(n_epochs):
      print(" processing epoch {}".format(current_epoch))
      
      t = time.process_time()
      e1,e3=0,0
      for u,i,r in train_df.values:


        x+=1
        #Just for saving checkpoints
        if x % 5000000==0:
          with open('/content/gdrive/MyDrive/svdpp/y/y_'+str(int(iteration+x/5000000))+'.npy', 'wb') as f:
            np.save(f, y)

          with open('/content/gdrive/MyDrive/svdpp/p/p_'+str(iteration+int(x/5000000))+'.npy', 'wb') as f:
            np.save(f, p)

          with open('/content/gdrive/MyDrive/svdpp/q/q_'+str(iteration+int(x/5000000))+'.npy', 'wb') as f:
            np.save(f, q)

          with open('/content/gdrive/MyDrive/svdpp/bu/bu_'+str(iteration+int(x/5000000))+'.npy', 'wb') as f:
            np.save(f, bu)

          with open('/content/gdrive/MyDrive/svdpp/bi/bi_'+str(iteration+int(x/5000000))+'.npy', 'wb') as f:
            np.save(f, bi)

        #Verbose
        if x % 100000==0:
          
          e=time.process_time()-t
          print("Batch:{} \t Total:{:.2f} | Lookup {:.2f} | sgd {:.2f}".format(int(x/100000),e,e1,e3))
          t=time.process_time()
          e1,e3=0,0




        u_index=m.profile_dict[u]
        i_index=m.asset_dict[i]

        
        try:
          #Nu = list(train_df[train_df['id_profile']==u]['id_asset'])
          t1=time.process_time()
          Nu=indexed_df.loc[u]['id_asset'].tolist()
          e1+=time.process_time()-t1
        except:
          print(u,i,r)


        #Getting user's implicit preference (Nu)
        Nu_index = m.get_many(Nu,'asset')
        
        I_Nu = len(Nu_index)
        sqrt_N_u = np.sqrt(I_Nu)
        y_u = np.sum(y[Nu_index], axis=0)
        u_impl_prf = y_u / sqrt_N_u



        rp = mu + bu[u_index] + bi[i_index] + np.dot(q[i_index], p[u_index] + u_impl_prf) 
        e_ui = r - rp

        #Gradient Descent
        bu[u_index] += lr * (e_ui - reg * bu[u_index])
        bi[i_index] += lr * (e_ui - reg * bi[i_index])
        p[u_index] += lr * (e_ui * q[i_index] - reg * p[u_index])
        q[i_index] += lr * (e_ui * (p[u_index] + u_impl_prf) - reg * q[i_index])
        t3=time.process_time()
        y[Nu_index] += lr * (e_ui * q[Nu_index] / sqrt_N_u - reg * y[Nu_index])
        ''' for j in Nu_index:
          y[j] += lr * (e_ui * q[j] / sqrt_N_u - reg * y[j]) '''
        e3+=time.process_time()-t3
      
  
    
    self.mu = mu
    self.bu =bu
    self.bi = bi
    self.p = p
    self.q=q
    self.y=y