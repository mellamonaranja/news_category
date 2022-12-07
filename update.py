import logging
import os
import re
import time
from logging import handlers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import hydra
import numpy as np
import pandas as pd
import pymysql
import tensorflow as tf
from transformers import AutoTokenizer

from models.MainModels import EncoderModel

# definition
# tensorflow 메모리 증가를 허용
gpus = tf.config.experimental.list_physical_devices("GPU")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# log setting
carLogFormatter = logging.Formatter("%(asctime)s,%(message)s")

carLogHandler = handlers.TimedRotatingFileHandler(
    filename="./log/predict.log",
    when="midnight",
    interval=1,
    encoding="utf-8",
)
carLogHandler.setFormatter(carLogFormatter)
carLogHandler.suffix = "%Y%m%d"

scarp_logger = logging.getLogger()
scarp_logger.setLevel(logging.INFO)
scarp_logger.addHandler(carLogHandler)


def processing(content):
    result = re.sub(r"[a-zA-Z가-힣]+뉴스", "", str(content))
    result = re.sub(r"[a-zA-Z가-힣]+ 뉴스", "", result)
    result = re.sub(r"[a-zA-Z가-힣]+newskr", "", result)
    result = re.sub(r"[a-zA-Z가-힣]+Copyrights", "", result)
    result = re.sub(r"[a-zA-Z가-힣]+ Copyrights", "", result)
    result = re.sub(r"\s+Copyrights", "", result)
    result = re.sub(r"[a-zA-Z가-힣]+com", "", result)
    result = re.sub(r"[가-힣]+ 기자", "", result)
    result = re.sub(r"[가-힣]+기자", "", result)
    result = re.sub(r"[가-힣]+ 신문", "", result)
    result = re.sub(r"[가-힣]+신문", "", result)
    result = re.sub(r"데일리+[가-힣]", "", result)
    result = re.sub(r"[가-힣]+투데이", "", result)
    result = re.sub(r"[가-힣]+미디어", "", result)
    result = re.sub(r"[가-힣]+ 데일리", "", result)
    result = re.sub(r"[가-힣]+데일리", "", result)
    result = re.sub(r"[가-힣]+ 콘텐츠 무단", "", result)
    result = re.sub(r"전재\s+변형", "전재", result)
    result = re.sub(r"[가-힣]+ 전재", "", result)
    result = re.sub(r"[가-힣]+전재", "", result)
    result = re.sub(r"[가-힣]+배포금지", "", result)
    result = re.sub(r"[가-힣]+배포 금지", "", result)
    result = re.sub(r"\s+배포금지", "", result)
    result = re.sub(r"\s+배포 금지", "", result)
    result = re.sub(r"[a-zA-Z가-힣]+.kr", "", result)
    result = re.sub(r"/^[a-z0-9_+.-]+@([a-z0-9-]+\.)+[a-z0-9]{2,4}$/", "", result)
    result = re.sub(r"[\r|\n]", "", result)
    result = re.sub(r"\[[^)]*\]", "", result)
    result = re.sub(r"\([^)]*\)", "", result)
    result = re.sub(r"[^ ㄱ-ㅣ가-힣A-Za-z0-9]", "", result)
    result = (
        result.replace("뉴스코리아", "")
        .replace("및", "")
        .replace("Copyright", "")
        .replace("저작권자", "")
        .replace("ZDNET A RED VENTURES COMPANY", "")
    )
    result = result.strip()

    return result


# label
labels = {
    "0": "인공지능",
    "1": "로봇",
    "2": "스마트팜",
    "3": "에너지",
    "4": "서버",
    "5": "투자",
    "6": "정부지원",
    "7": "증강현실",
    "8": "이동수단",
    "9": "개발",
    "10": "통신",
    "11": "과학",
    "12": "드론",
    "13": "블록체인",
    "14": "핀테크",
    "15": "커머스",
    "16": "여행",
    "17": "미디어",
    "18": "헬스케어",
    "19": "의약",
    "20": "식품",
    "21": "교육",
    "22": "직업",
    "23": "경제",
    "24": "광고",
    "25": "제약",
    "26": "O2O",
    "27": "뷰티",
    "28": "부동산",
    "29": "etc",
}


def data_load():
    print("dataload start")
    conn = pymysql.connect(
        user="root",
        passwd="Illunex123!",
        db="portal_news_scraper",
        host="172.30.1.100",
        port=3306,
        charset="utf8",
        use_unicode=True,
    )
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    query = """select id, content, create_date from portal_news_scraper.portal_news where predict is null limit 100"""

    cursor.execute(query)
    data = pd.DataFrame(cursor.fetchall(), columns=["id", "content", "create_date"])
    data = data[["id", "content", "create_date"]]

    data["content"] = data.content.apply(processing)
    df = data.drop_duplicates()
    print("dataload end")

    return df


def update(query, param):
    conn = pymysql.connect(
        user="root",
        passwd="Illunex123!",
        db="portal_news_scraper",
        host="172.30.1.100",
        port=3306,
        charset="utf8",
        use_unicode=True,
    )
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    cursor.executemany(query, param)
    conn.commit()


@hydra.main(config_name="config.yml")
def predict(cfg):
    try:
        # start time
        start = time.time()
        print("start : " + str(start))

        # dataload
        df = data_load()

        tokenizer = AutoTokenizer.from_pretrained(cfg.ETC.tokenizer_dir)
        model = EncoderModel.load(cfg.ETC.output_dir)
        data = tokenizer(
            df["content"].to_list(),
            max_length=model.seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="tf",
        )

        print("start predict")
        pred = model.predict(dict(data))

        label = [
            [j for j in r if pred[i, j] >= 0.5]
            for i, r in enumerate(np.argsort(pred)[:, :-4:-1])
        ]
        df["label"] = label
        df["predict"] = df.label.apply(
            lambda x: ", ".join(labels.get(str(e)) for e in x)
        )

        # dataframe to update query
        query = "update portal_news_scraper.portal_news set predict=%s where id=%s;"
        param = []
        for i in range(len(df)):
            temp = (df["predict"][i], df["id"][i])
            param.append(temp)

        update(query, param)

        # end time
        logging.info("time :" + str(time.time() - start))

    except Exception as e:
        logging.info(e)
        return 200


if __name__ == "__main__":
    predict()
