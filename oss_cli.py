import io
import os
from queue import Queue

import boto3
from botocore.config import Config
from pydub import AudioSegment
import torchaudio
from tqdm import tqdm


class OSSClient:
    def __init__(self, endpoint_url, access_key, secret_key):
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key
        self.client = boto3.client('s3', endpoint_url=endpoint_url,
                                   aws_access_key_id=access_key,
                                   aws_secret_access_key=secret_key,
                                   config=Config(proxies={'http': None, 'https': None})
                                )

    def get_file(self, bucket_name, filename):
        data = self.client.get_object(Bucket=bucket_name, Key=filename)
        return data["Body"]

    def put_file(self, bucket_name, file_path, buffer, data_type="audio/flac"):
        self.client.put_object(Bucket=bucket_name, Key=file_path, Body=buffer, ContentType=data_type)

    def exists(self, bucket_name, file_path):
        try:
            return self.client.head_object(Bucket=bucket_name, Key=file_path)
        except Exception as e:
            return False

    def list_files(self, bucket_name, file_path, delimiter=None, continuation_token=None, max_keys=1000):
        if continuation_token:
            response = self.client.list_objects_v2(Bucket=bucket_name, Prefix=file_path, Delimiter=delimiter, ContinuationToken=continuation_token, MaxKeys=max_keys)
        else:
            response = self.client.list_objects_v2(Bucket=bucket_name, Prefix=file_path, Delimiter=delimiter, MaxKeys=max_keys)
        return response

    def get_all_files(self, bucket_name, prefix, count=None):
        files = []
        # 使用分页器处理大量文件
        paginator = self.client.get_paginator('list_objects_v2')
        for page in tqdm(paginator.paginate(Bucket=bucket_name, Prefix=prefix), desc="paginate"):
            if 'Contents' in page:
                for obj in page['Contents']:
                    files.append(obj['Key'])
                    if count is not None and len(files) > count:
                        return files
        return files


class OSSPool:
    def __init__(self, max_conn=16, datatype="B"):
        self.pool = Queue(max_conn)
        if datatype == "B":
            for _ in range(max_conn):
                oss_client = OSSClient(endpoint_url="http://oss.i.shaipower.com",
                                       access_key="d6562efa47b2c0d1ca72a1890de2337b",
                                       secret_key="4bff2addfe97368cdf02134e60eadcb0")
                self.pool.put(oss_client)  # 预创建连接
        else:
            for _ in range(max_conn):
                oss_client = OSSClient(endpoint_url="http://oss.i.basemind.com",
                                       access_key="883cf6043ca07c05bcfc6b1981f8030b",
                                       secret_key="ef4fee8fa77914fe9eb93ad005603cc1")
                self.pool.put(oss_client)  # 预创建连接
        assert self.pool is not None

    def get_conn(self):
        return self.pool.get()

    def release_conn(self, conn):
        self.pool.put(conn)

OSS_POOL = OSSPool(datatype="B")

def read_audio(filename: str):
    if filename.startswith("s3://"):
        oss_client = OSS_POOL.get_conn()
        bucket_name = filename[5:].split("/", 1)[0]
        filename = filename[5:].split("/", 1)[1]
        wavdata = oss_client.get_file(bucket_name, filename).read()
        OSS_POOL.release_conn(oss_client)
        audio_io = io.BytesIO(wavdata)
        audio, in_sr = torchaudio.load(audio_io)
    else:
        ext = filename.split(".")[-1]

        audio, in_sr = torchaudio.load(filename, format=ext)

    return audio

if __name__ == "__main__":
    pass