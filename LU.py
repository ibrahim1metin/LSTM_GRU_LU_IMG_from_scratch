import tensorflow as tf
class LU(tf.keras.layers.Layer):
    def __init__(self,input_nodes,hid_nodes):
        self.input_nodes=input_nodes
        self.hid_nodes=hid_nodes
    def build(self):
        self.ux=self.add_weight(shape=(self.hid_nodes,self.hid_nodes),trainable=True)
        self.wx=self.add_weight(shape=(self.input_nodes,self.hid_nodes),trainable=True)
        self.wx2=self.add_weight(shape=(self.input_nodes,self.hid_nodes),trainable=True)
        self.bx2=self.add_weight(shape=(self.hid_nodes,),trainable=True)
        self.bx=self.add_weight(shape=(self.hid_nodes,),trainable=True)
        self.uh=self.add_weight(shape=(self.hid_nodes,self.hid_nodes),trainable=True)
    def call(self,state,input):
        u=tf.math.sigmoid(tf.matmul(input,self.wx)+tf.matmul(state,self.ux)+self.bx)
        h_=tf.math.tanh(tf.matmul(input,self.wx2)+self.bx2+tf.matmul(state,self.uh))
        h=tf.multiply((1-u),state)+tf.multiply(u,h_)
        return h
