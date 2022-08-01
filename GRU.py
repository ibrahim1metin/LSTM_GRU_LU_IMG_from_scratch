import tensorflow as tf
class GRU(tf.keras.layers.Layer):
    def __init__(self,inp_nod,hid_nod):
        self.input_nodes=inp_nod
        self.hidden_nodes=hid_nod
    def build(self):
        self.wx=self.add_weight(shape=(self.input_nodes,self.hidden_nodes),trainable=True)
        self.bx=self.add_weight(shape=(self.hidden_nodes,),trainable=True)
        self.wh=self.add_weight(shape=(self.hidden_nodes,self.hidden_nodes),trainable=True)
        self.wr=self.add_weight(shape=(self.input_nodes,self.hidden_nodes),trainable=True)
        self.ur=self.add_weight(shape=(self.hidden_nodes,self.hidden_nodes),trainable=True)
        self.uz=self.add_weight(shape=(self.hidden_nodes,self.hidden_nodes),trainable=True)
        self.wz=self.add_weight(shape=(self.input_nodes,self.hidden_nodes),trainable=True)
        self.br=self.add_weight(shape=(self.hidden_nodes,),trainable=True)
        self.bz=self.add_weight(shape=(self.hidden_nodes,),trainable=True)
    def call(self,state,inp):
        z=tf.math.sigmoid(tf.linalg.matmul(inp,self.wz)+tf.linalg.matmul(state,self.uz)+self.bz)
        r=tf.math.sigmoid(tf.linalg.matmul(inp,self.wr)+tf.linalg.matmul(state,self.ur)+self.br)
        h_=tf.math.tanh(tf.linalg.matmul(inp,self.wx)+self.bx+tf.linalg.matmul(state,self.wh)*r)
        hid_state=tf.math.multiply((1-z),h_)+tf.math.multiply(state,z)
        return hid_state
