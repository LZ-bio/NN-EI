#from keras import backend as K
import tensorflow as tf
from tensorflow.keras import backend as K

#def focal_loss1(alpha=0.75, gamma=2.0):
#    def focal_loss_fixed(y_true, y_pred):
        # y_true 是个一阶向量, 下式按照加号分为左右两部分
        # 注意到 y_true的取值只能是 0或者1 (假设二分类问题)，可以视为“掩码”
        # 加号左边的 y_true*alpha 表示将 y_true中等于1的槽位置为标量 alpha
        # 加号右边的 (ones-y_true)*(1-alpha) 则是将等于0的槽位置为 1-alpha
#        ones = K.ones_like(y_true)
#        alpha_t = y_true*alpha + (ones-y_true)*(1-alpha)

        # 类似上面，y_true仍然视为 0/1 掩码
        # 第1部分 `y_true*y_pred` 表示 将 y_true中为1的槽位置为 y_pred对应槽位的值
        # 第2部分 `(ones-y_true)*(ones-y_pred)` 表示 将 y_true中为0的槽位置为 (1-y_pred)对应槽位的值
        # 第3部分 K.epsilon() 避免后面 log(0) 溢出
#        p_t = y_true*y_pred + (ones-y_true)*(ones-y_pred) + K.epsilon()

        # 就是公式的字面意思
#        focal_loss = -alpha_t * K.pow((ones-p_t),gamma) * K.log(p_t)
#    return focal_loss_fixed

def focal_loss(alpha=0.75, gamma=3.0):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred,epsilon,1.0-epsilon)
        y_true = tf.cast(y_true,tf.float32)
        alpha_t = y_true*alpha + (1-y_true)*(1-alpha)
        p_t = y_true*y_pred + (1-y_true)*(1-y_pred)
        focal_loss = -alpha_t * K.pow((1-p_t),gamma) * K.log(p_t)
        return K.mean(focal_loss)
    return focal_loss_fixed

# taken from https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


from tensorflow.keras import backend as K

def R_set(x):
    n_sample = K.shape(x)[0]  # Get the batch size dimension
    matrix_ones = K.ones((n_sample, n_sample))  # Create ones matrix
    indicator_matrix = K.tf.linalg.band_part(matrix_ones, -1, 0)  # Create lower triangular matrix
    return indicator_matrix


def neg_par_log_likelihood(pred, ytime, yevent):
    n_observed = K.sum(yevent, axis=0)
    ytime_indicator = R_set(ytime)  # Assuming R_set is already adapted for TensorFlow
    #if len(K.tensorflow_backend._get_available_gpus()) > 0:
    #    ytime_indicator = K.tensorflow_backend.identity(ytime_indicator)  # This ensures it's on GPU if available
    risk_set_sum = K.dot(ytime_indicator, K.exp(pred))
    diff = pred - K.log(risk_set_sum)
    sum_diff_in_observed = K.dot(K.transpose(diff), yevent)
    cost = (- (sum_diff_in_observed / n_observed))
    cost = K.reshape(cost, (-1,))
    return cost


from lifelines.utils import concordance_index
def evaluate_survival(y_true,y_pred):
    event_time = y_true[:, 0]
    event_indicator = y_true[:, 1]
    c_index = concordance_index(event_time,y_pred,event_indicator)
    return c_index


def cox_loss(y_true, y_pred):
    """
    Cox比例风险模型损失函数
    y_true: 包含事件时间(event time)和事件指示(event indicator)的张量
    y_pred: 模型预测的风险分数
    """
    event_time = y_true[:, 0]
    event_indicator = y_true[:, 1]
    
    # 创建风险集矩阵(R矩阵)
    n = tf.shape(event_time)[0]
    R = tf.tile(tf.expand_dims(event_time, 1), [1, n])
    R = tf.cast(R >= tf.transpose(R), tf.float32)
    
    # 计算风险集总和
    exp_pred = tf.exp(y_pred)
    risk_set_sum = tf.matmul(R, exp_pred)
    
    # 计算部分对数似然
    diff = y_pred - tf.math.log(risk_set_sum)
    loss = -tf.reduce_sum(diff * event_indicator) / tf.reduce_sum(event_indicator)
    
    return loss


def binary_cross_entropy(recon_x, x):
    """
    TensorFlow 版本的二元交叉熵损失函数
    参数:
        recon_x: 重构的预测值 (形状: [batch_size, ...])
        x: 原始输入值 (形状: 需与 recon_x 相同)
    返回:
        每个样本的损失值 (形状: [batch_size])
    """
    return -tf.reduce_sum(
        x * tf.math.log(recon_x + 1e-8) + 
        (1 - x) * tf.math.log(1 - recon_x + 1e-8), 
        axis=-1
    )


# 计算总损失
def loss_Likelihood(recon_x, x):
    """
    计算似然损失，与原始PyTorch实现一致
    参数:
        recon_x: 重构的预测值
        x: 原始输入值
    返回:
        标量损失值
    """
    bce = binary_cross_entropy(recon_x, x)
    return tf.reduce_sum(bce) / tf.cast(tf.shape(x)[0], tf.float32)


def weight_binary_cross_entropy(recon_x, x,pos_weight=1):
    """
    TensorFlow 版本的二元交叉熵损失函数
    参数:
        recon_x: 重构的预测值 (形状: [batch_size, ...])
        x: 原始输入值 (形状: 需与 recon_x 相同)
    返回:
        每个样本的损失值 (形状: [batch_size])
    """
    return -tf.reduce_sum(
        x * tf.math.log(recon_x + 1e-8)*pos_weight + 
        (1 - x) * tf.math.log(1 - recon_x + 1e-8), 
        axis=-1
    )


# 计算总损失
def loss_Likelihood_weight(recon_x, x,pos_weight=100):
    """
    计算似然损失，与原始PyTorch实现一致
    参数:
        recon_x: 重构的预测值
        x: 原始输入值
    返回:
        标量损失值
    """
    bce = weight_binary_cross_entropy(recon_x, x,pos_weight=pos_weight)
    return tf.reduce_sum(bce) / tf.cast(tf.shape(x)[0], tf.float32)