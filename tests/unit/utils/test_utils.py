import os


def find_user_rows(rows, user_id, field="USER_ID"):
    return [r for r in rows if r[field] == user_id]

def mkdirs_mock(path):
    dir_ = os.path.dirname(path)
    if not os.path.exists(dir_):
        mkdirs_mock(dir_)
    os.mkdir(path)

def file_exists_mock(path, dbutils):
    try:
        dbutils.fs.ls(path)
        return True
    except FileNotFoundError:
        return False

