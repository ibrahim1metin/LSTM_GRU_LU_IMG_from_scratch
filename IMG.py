import tensorflow as tf
class IMG(tf.keras.layers.Layer):
    def __init__(self,hid_nodes,**kwargs):
        super(IMG,self).__init__(dynamic=True)
        self.hid_nodes=hid_nodes
        self.state_size=[tf.TensorShape((hid_nodes,)),tf.TensorShape((hid_nodes,))]
        self.output_size=hid_nodes
    def build(self,i):
        self.wh=self.add_weight(shape=(i[-1],self.hid_nodes),trainable=True)
        self.uh=self.add_weight(shape=(self.hid_nodes,self.hid_nodes),trainable=True)
        self.bh=self.add_weight(shape=(self.hid_nodes,),trainable=True)
        self.wp=self.add_weight(shape=(i[-1],self.hid_nodes),trainable=True)
        self.up=self.add_weight(shape=(self.hid_nodes,self.hid_nodes),trainable=True)
        self.gp=self.add_weight(shape=(self.hid_nodes,self.hid_nodes),trainable=True)
        self.bp=self.add_weight(shape=(self.hid_nodes,),trainable=True)
    def call(self,input,states):
        prev_p,prev_s=states[0],states[1]
        p=tf.sigmoid(tf.matmul(input,self.wp)+tf.matmul(prev_s,self.up)+tf.matmul(prev_p,self.gp)+self.bp)
        h_=tf.tanh(tf.matmul(input,self.wh)+tf.matmul(prev_s,self.uh)+self.bh)
        h=tf.multiply((1-p),prev_s)+tf.multiply(p,h_)
        return h,[[p,h]]
    def get_config(self):
        config=super().get_config()
        config.update({"hid_nodes":self.hid_nodes})
        return config
