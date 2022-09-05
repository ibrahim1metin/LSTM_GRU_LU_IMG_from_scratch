import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
class LU(tf.keras.layers.Layer):
    def __init__(self,hid_nodes,dynamic=True):
        super(LU,self).__init__(dynamic=dynamic)
        self.hid_nodes=hid_nodes
        self.state_size=hid_nodes
    def build(self,i):
        self.ux=self.add_weight(shape=(self.hid_nodes,self.hid_nodes),trainable=True)
        self.wx=self.add_weight(shape=(i[-1],self.hid_nodes),trainable=True)
        self.wx2=self.add_weight(shape=(i[-1],self.hid_nodes),trainable=True)
        self.bx2=self.add_weight(shape=(self.hid_nodes,),trainable=True)
        self.bx=self.add_weight(shape=(self.hid_nodes,),trainable=True)
        self.uh=self.add_weight(shape=(self.hid_nodes,self.hid_nodes),trainable=True)
        self.built=True
    @tf.function()
    def call(self,input,state):
        u=tf.math.sigmoid(tf.matmul(input,self.wx)+tf.matmul(state,self.ux)+self.bx)
        h_=tf.math.tanh(tf.matmul(input,self.wx2)+self.bx2+tf.matmul(state,self.uh))
        h=tf.multiply((1-u),state)+tf.multiply(u,h_)
        return h
