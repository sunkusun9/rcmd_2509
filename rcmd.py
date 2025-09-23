import tensorflow as tf
import numpy as np
from multiprocessing import Process, Queue

class EmbModel(tf.keras.Model):
    def __init__(self, ids, rank, l2=0):
        """
        임베딩 
        Parameters: 
            ids: list
                카테고리의 수준들의 id 리스트
            rank: int
                임베딩 벡터의 사이즈
            l2: float
                l2 규제 계수
        """
        super().__init__()
        self.lu_ids = tf.keras.layers.IntegerLookup(vocabulary=tf.constant(ids))
        
        if l2 > 0:
            reg = tf.keras.regularizers.L2(l2)
        else:
            reg = None
        self.emb = tf.keras.layers.Embedding(len(ids) + 1, rank, embeddings_regularizer=reg)
    
    def call(self, x, training=False):
        x = self.lu_ids(x)
        return self.emb(x, training=training)
    
class MeanModel(tf.keras.Model):
    def __init__(self, mean, user_mean_model, movie_mean_model):
        super().__init__()
        self.mean = tf.constant([mean], dtype=tf.float32)
        self.user_mean_model = user_mean_model
        self.movie_mean_model = movie_mean_model
    
    def call(self, x, training=False):
        return self.mean + self.user_mean_model(x['userId'], training=training) + \
            self.movie_mean_model(x['movieId'], training=training)
    
# 제공한 모델 각각의 예측 결과를 더하는 모델을 만듭니다.
class AdditiveModel(tf.keras.Model):
    def __init__(self, models):
        """
        Parameters:
            models: list
                tf.keras.Model 객체로 이루진 리스트입니다.
        """
        super().__init__()
        self.models = models
    
    def call(self, x, training=False):
        # 각각의 모델에서 나온 출력을 모으기 위한 리스트 입니다.
        y_hat = []
        for i in self.models:
            y_hat.append(i(x, training=training))
        return tf.reduce_sum(y_hat, axis=0)
    
class UserHistModel(tf.keras.Model):
    """
    사용자가 이전에 평가한 영화와 평점을 입력 받는 모델입니다.
    """
    def __init__(self, user_model, movie_model, rank, l2=0):
        """
        Parameters
            user_model: tf.keras.Model
                사용자 모델
            movie_model: tf.keras.Model
                영화 모델
            rank: int
                출력 벡터의 수
            l2: float
                L2 규제, 0일 때는 규제를 사용하지 않습니다.
        """
        super().__init__()
        self.user_model = user_model
        self.movie_model = movie_model
        
        if l2 > 0:
            reg = tf.keras.regularizers.L2(l2)
        else:
            reg = None
        
        # Rank 벡터를 만들어 내기 위한 밀집 신경망을 구성합니다. 
        # 첫번째 은닉층(1st Hidden Layer )
        self.dl = tf.keras.layers.Dense(
            64, activation='relu', 
            kernel_initializer=tf.keras.initializers.HeNormal(), # relu와 같은 0이상의 값만을 갖는 활성화함수는 초기화 방밥에는 HeNormal이 유용
            kernel_regularizer=reg
        )
        
        # 출력층(Output Layer)
        self.o = tf.keras.layers.Dense(
            rank, 
            kernel_regularizer=reg
        )
        
        # 사용자 벡터, 이전 시청 영화 벡터, 평점을 결합하기 위한 결합층(Concatenate Layer)을 생성합니다.
        self.cc = tf.keras.layers.Concatenate(axis=-1)
    
    def call(self, x, prev_movieId, prev_rating, training=False):
        vec = self.cc([
            self.user_model(x, training=training), # 사용자 벡터를 가져옵니다. N×rank
            self.movie_model(prev_movieId, training=training), # 이전 시청 영화 벡터를 가져옵니다. N×rank
            tf.expand_dims(
                prev_rating, # N
                axis=-1
            ) # 이전 평점. N×1
        ]) # N×(2×rank + 1)
        
        vec = self.dl(vec) # 첫번째 은닉층. N×64
        return self.o(vec) # 출력층. N×rank
    
class MovieInfoModel(tf.keras.Model):
    def __init__(self, df_movieinfo, movie_model, emb_config, rank, l2=0):
        super().__init__()
        self.lu_movie = tf.keras.layers.IntegerLookup(
            vocabulary=df_movieinfo.index[1:].values
        )
        # 영화 별 장르 정보를 담고 저장공간생성
        cols_genre = [i for i in df_movieinfo.columns if i.startswith('genre')]
        self.genres = tf.constant(
            df_movieinfo[cols_genre]
        )
        # 영화의 컬렉션 정보를 지니고 있는 저장공간 생성
        self.collection = tf.constant(
            df_movieinfo['collection']
        )
        # 영화의 제목 + 줄거리의 OpenAI에서 구한 Embedding 정보 저장공간 생성
        self.ov_emb = tf.constant(
            df_movieinfo['ov_emb'].tolist()
        )
        self.release_date = tf.constant(df_movieinfo['release_date'], dtype=tf.int64)
        self.movie_model = movie_model
        if l2 > 0:
            reg = tf.keras.regularizers.L2(l2)
        else:
            reg = None
        self.emb_collection = tf.keras.layers.Embedding(
            df_movieinfo['collection'].max() + 1, 
            emb_config['collection'], 
            embeddings_regularizer = reg
        )

        # 은닉레이어 1: Dense(64)
        self.dl = tf.keras.layers.Dense(
            64, activation='relu', 
            kernel_initializer=tf.keras.initializers.HeNormal(), 
            kernel_regularizer=reg
        )
        # 출력 레이어: Dense(rank)
        self.o = tf.keras.layers.Dense(rank)
        # 결합 레이어
        self.cc = tf.keras.layers.Concatenate(axis = -1)
        self.rank = rank
    
    def call(self, x, ts, training=False):
        x_movie = self.movie_model(x, training=training)
        
        x = self.lu_movie(x)
        x_genre = tf.gather(self.genres, x) # 장르 여부를 가져옵니다.
        x_collection = tf.gather(self.collection, x) # 컬렉션 번호 가져옵니다.
        x_collection = self.emb_collection(x_collection) # 컬렉션 임베딩을 가져옵니다.
        x_ov_emb =  tf.gather(self.ov_emb, x) # OpenAI API를 얻은 영화 줄거리의 Embedding 정보를 가져 옵니다.
        x_ts = tf.gather(self.release_date, x)
        x_ts = tf.cast(
            (tf.minimum(
                tf.math.log(
                    tf.maximum(
                        tf.cast(ts  - x_ts, tf.float64) // 1e9 // 3600 // 24, 1
                    )
                ), 10
            ) - 5) / 5, tf.float32
        )
        x =  self.cc([
            x_movie, x_genre, x_collection, x_ov_emb, tf.expand_dims(x_ts, axis = -1)
        ]) # x_movie, x_genre, x_collection, x_ov_emb를 결합하여 하나의 텐서로 만듭니다.
        x = self.dl(x)
        return self.o(x)

class UserHistModel2(tf.keras.Model):
    def __init__(self, user_model, movie_model, rank, l2 = 0, rnn = "lstm"):
        """
        Parameters
            user_model: tf.keras.Model
                사용자 모델
            movie_model: tf.keras.Model
                영화 모델
            rank: int
                출력 벡터의 수
            l2: float
                L2 규제, 0일 때는 규제를 사용하지 않습니다.
            rnn: str
                시청이력에서 사용할 RNN 종류
        """
        super().__init__()
        self.user_model = user_model
        self.movie_model = movie_model
        if rnn == "lstm":
            self.rnn = tf.keras.layers.LSTM(32)
        elif rnn == "gru":
            self.rnn = tf.keras.layers.GRU(32)
        else:
            self.rnn = None
        if l2 > 0:
            reg = tf.keras.regularizers.L2(l2)
        else:
            reg = None
        self.dl = tf.keras.layers.Dense(
            64, activation='relu', 
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer=reg
        )
        self.o = tf.keras.layers.Dense(
            rank,
            kernel_regularizer=reg
        )
        self.cc = tf.keras.layers.Concatenate(axis=-1)
        self.cc2 = tf.keras.layers.Concatenate(axis=-1)

    def call(self, x, prev_movieIds, prev_ratings, prev_ts, training=False):
        hist_vec = self.cc2([
            self.movie_model(prev_movieIds, prev_ts, training=training), 
            prev_ratings
        ])
        if self.rnn != None:
            hist_vec = self.rnn(hist_vec, training=training)
        else:
            hist_vec = tf.reduce_mean(hist_vec, axis= -2)
        vec = self.cc([
            self.user_model(x, training=training), 
            hist_vec
        ])
        vec = self.dl(vec)
        return self.o(vec)
    
class BatchWoker(Process):
    def __init__(self, q, df, df_hist, pos, hist_cnt, batch_size, default_rating):
        self.q = q
        self.hist_cnt = hist_cnt
        self.batch_size = batch_size
        self.df = df
        self.df_hist = df_hist
        self.pos = pos
        self.z = np.zeros(hist_cnt, dtype=np.int64)
        self.o = np.ones(hist_cnt, dtype=np.int64)
        self.rating_m = np.ones(hist_cnt) * default_rating
        super().__init__()
    def run(self):
        for i in self.pos:
            piece = self.df.iloc[i:i + self.batch_size]
            hist = self.df_hist.loc[piece['userId']]
            prev_movies, prev_ratings, prev_ts = list(), list(), list()
            # 해당 시점의 위치 인덱스를 찾고 시점의 hist_cnt만큼의 평가 이력을 가져옵니다.
            for hist_date, hist_movieId, hist_rating, hist_ts, d, ts in zip(hist['date'], hist['movieId'], hist['rating'], hist['ts'],piece['date'], piece['ts']):
                # 현재 평가시점에 해당하는 위치인덱스를 가져옵니다.
                idx = np.searchsorted(hist_date, d)
                # 가져올 평가 이력의 시작 지점을 계산합니다.
                from_idx  = max(0, idx - self.hist_cnt)
                null_hist_idx = max(0, self.hist_cnt - idx)
                if null_hist_idx > 0:
                    # 이전 평가 인덱스를 가져옵니다. 이전 이력이 없으면 Null Movie 0을 전달합니다.
                    prev_movies.append(np.hstack([self.z[:null_hist_idx], hist_movieId[from_idx:idx]]))
                    # 이전 평가 rating을 가져옵니다. 이전 이력이 없으면 평균 편점을 전달합니다.
                    prev_ratings.append(np.hstack([self.rating_m[:null_hist_idx], hist_rating[from_idx:idx]]))
                    # 이전 평가 ts을 가져옵니다. 마지막 평가 시점을 전달 합니다.
                    prev_ts.append(np.hstack([self.o[:null_hist_idx] * ts, hist_ts[from_idx:idx]]))
                else:
                    prev_movies.append(hist_movieId[from_idx:idx])
                    prev_ratings.append(hist_rating[from_idx:idx])
                    prev_ts.append(hist_ts[from_idx:idx])
            batch = (piece['userId'], piece['movieId'], piece['ts'], prev_movies, prev_ratings, prev_ts, piece['rating'])
            self.q.put(batch)
        self.q.put(None) # 할당량을 채웠을 경우에 None을 반환합니다.
        
def hist_set_iter(df, df_hist, hist_cnt, batch_size, pbar=None, shuffle=True, n_job=4, queue_size=16, default_rating=0.0, equal_batch_size=True):
    if len(df) % batch_size != 0 and equal_batch_size:
        # 전체셋 크기가 batch_size에 나누어 떨어지지 않으면
        # 동일한 Batch 사이즈를 위해 마지막은 끝에서 batch_size만큼 뺀 위치 부터 가져옵니다.
        pos = np.hstack([np.arange(0, len(df) - batch_size, batch_size), [len(df) - batch_size]])
    else:
        pos = np.arange(0, len(df), batch_size)
    if shuffle:
        np.random.shuffle(pos)
    chunk = (len(pos) + n_job - 1) // n_job
    # Work Process 에서 작업한 결과를 저장할 Queue입니다.
    q = Queue(queue_size)
    # n_jobs 만큼 Worker를 생성합니다.
    workers = list()
    for i in range(0, len(pos), chunk):
        workers.append(BatchWoker(q, df, df_hist, pos[i: i + chunk], hist_cnt, batch_size, default_rating))
        workers[-1].start()
    job_finished = 0
    if pbar is not None:
        pbar.total = len(pos)
    while(True):
        val = q.get()
        if val == None: # Worker가 작업을 마치면 None을 반환합니다.
            job_finished +=1 # 작업이 끝난 Worker 카운트
            if job_finished == len(workers): # 모든 Worker가 작업을 끝냈다면 중지
                break
            continue
        yield(val)
        if pbar is not None:
            pbar.update(1)