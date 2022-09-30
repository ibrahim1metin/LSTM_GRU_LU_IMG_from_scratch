import tensorflow as tf
class lstm(tf.keras.layers.Layer):
    def __init__(self,hidunits,**kwargs):
      super(lstm,self).__init__(dynamic=False)
      self.hidunits=hidunits
      self.state_size=[tf.TensorShape((hidunits,)),tf.TensorShape((hidunits,))]
      self.output_size=hidunits
    def build(self,i):
      self.inpunits=i[-1]
      self.weighti=self.add_weight(shape=(self.inpunits,self.hidunits),trainable=True)
      self.biasi=self.add_weight(shape=(self.hidunits,),trainable=True)
      self.rec_weighti=self.add_weight(shape=(self.hidunits,self.hidunits),trainable=True)

      self.weightf=self.add_weight(shape=(self.inpunits,self.hidunits),trainable=True)
      self.biasf=self.add_weight(shape=(self.hidunits,),trainable=True)
      self.rec_weightf=self.add_weight(shape=(self.hidunits,self.hidunits),trainable=True)

      self.weightg=self.add_weight(shape=(self.inpunits,self.hidunits),trainable=True)
      self.biasg=self.add_weight(shape=(self.hidunits,),trainable=True)
      self.rec_weightg=self.add_weight(shape=(self.hidunits,self.hidunits),trainable=True)

      self.weightc=self.add_weight(shape=(self.inpunits,self.hidunits),trainable=True)
      self.biasc=self.add_weight(shape=(self.hidunits,),trainable=True)
      self.rec_weightc=self.add_weight(shape=(self.hidunits,self.hidunits),trainable=True)
      self.built=True
    def call(self,inputs,states):
      prevstate,prevoutput=tf.unstack(states)
      i=tf.math.sigmoid(tf.linalg.matmul(inputs,self.weighti)+tf.linalg.matmul(prevstate,self.rec_weighti)+self.biasi)
      f=tf.math.sigmoid(tf.linalg.matmul(inputs,self.weightf)+tf.linalg.matmul(prevstate,self.rec_weightf)+self.biasf)
      g=tf.math.sigmoid(tf.linalg.matmul(inputs,self.weightg)+tf.linalg.matmul(prevstate,self.rec_weightg)+self.biasg)
      c_=tf.math.tanh(tf.linalg.matmul(inputs,self.weightc)+tf.linalg.matmul(prevstate,self.rec_weightc)+self.biasc)
      c=f*prevoutput+i*c_
      state=g*tf.math.tanh(c)
      return c,tf.stack([state,c])
    def get_config(self):
        config=super().get_config()
        config.update({"hidunits":self.hidunits})
        return config
