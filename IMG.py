import tensorflow as tf
class IMG(tf.keras.layers.Layer):
    def __init__(self,input_nods,hid_nods):
        self.input_nods=input_nods
        self.hid_nods=hid_nods
    def build(self):
        self.wh=self.add_weight(shape=(self.hid_nods,self.hid_nods),trainable=True)
        self.uh=self.add_weight(shape=(self.hid_nods,self.hid_nods),trainable=True)
        self.bh=self.add_weight(shape=(self.hid_nods,),trainable=True)

        self.wp=self.add_weight(shape=(self.input_nods,self.hid_nods),trainable=True)
        self.up=self.add_weight(shape=(self.hid_nods,self.hid_nods),trainable=True)
        self.gp=self.add_weight(shape=(self.hid_nods,self.hid_nods),trainable=True)
        self.bp=self.add_weight(shape=(self.hid_nods,self.hid_nods),trainable=True)
    def call(self,states,input):
        prev_p,prev_s=tf.unstack(states)
        p=tf.sigmoid(tf.matmul(input,self.wp)+tf.matmul(prev_s,self.up)+tf.matmul(prev_p,self.gp)+self.bp)
        h_=tf.tanh(tf.matmul(input,self.wh)+tf.matmul(prev_s,self.uh)+self.bh)
        h=tf.multiply((1-p),prev_s)+tf.multiply(p,h_)
        return tf.stack([p,h])
