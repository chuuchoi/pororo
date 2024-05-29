from ftplib import FTP
import time
import os


def list_recursive(ftp, remotedir):
    ftp.cwd(remotedir)
    for entry in ftp.mlsd():
        if entry[1]['type'] == 'dir':
            remotepath = remotedir + "/" + entry[0]
            #print(time.time(), remotepath)
            list_recursive(ftp, remotepath)
        else:
            print(entry)

def download_files_recursively(ftp, remote_path, local_path):
    # 현재 폴더 내 파일 목록 가져오기
    file_list = []
    ftp.mlsd("./", file_list.append)
    #ftp.retrlines('LIST', file_list.append) # 디렉토리의 내용을 목록화

    for item in file_list:
        file_name, file_info = item
        remote_item_path = os.path.join(remote_path, file_name)
        local_item_path = os.path.join(local_path, file_name)

        if file_info["type"] == "dir":
            # 폴더인 경우 재귀적으로 폴더 내 파일 다운로드
            if not os.path.exists(local_item_path):
                os.makedirs(local_item_path)
            download_files_recursively(ftp, remote_item_path, local_item_path)
        else:
            # 파일인 경우 다운로드
            with open(local_item_path, "wb") as local_file:
                ftp.retrbinary("RETR " + remote_item_path, local_file.write)
                print("ftp path : " + remote_item_path)

def main():
    # FTP 서버 접속 정보 입력 받기
    host = "data.gamecrewlab.com"
    username = "djdmsehd"
    password = "Localhost@12"
    folder_path = "/HDD1/2023_NIA_78/Test"
    save_dir_path = r"E:\NIA78번-관광데이터\데이터셋\원천데이터"

    # 로컬 다운로드 폴더 생성
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    # FTP 서버에 접속
    ftp = FTP()
    ftp.connect(host, 8082);
    ftp.login(username, password)

    # 입력받은 폴더로 이동
    ftp.cwd(folder_path)

    # 재귀적으로 파일 다운로드
    #download_files_recursively(ftp, folder_path, save_dir_path)
    list_recursive(ftp, folder_path)

    # FTP 연결 종료
    ftp.quit()

    print("파일 다운로드 완료")

if __name__ == "__main__":
    main()
