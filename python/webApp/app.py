
from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return "index"
# *******************************************************
"""
1. 装换类型
string  （缺省值） 接受任何不包含斜杠的文本
int     接受正整数
float   接受正浮点数
path    类似 string ，但可以包含斜杠
uuid    接受 UUID 字符串

路由的响应类型为: string, dict, list, tuple with headers or status, Response instance, or WSGI callable
"""

@app.route("/string/<string:string_var>")
def get_string_from_requests(string_var):
    print(f"string_var:{string_var}, {type(string_var)}")
    return f"string_var:{string_var}"

@app.route("/int/<int:i_var>")
def get_int(i_var):
    return str(i_var)

@app.route("/float/<float:f_var>/")
def get_float(f_var):
    return str(f_var)

@app.route("/path/<path:p_var>")
def get_path(p_var):
    return p_var

@app.route("/uid/")
def get_uid():
    from uuid import uuid4
    uid = str(uuid4())
    return uid

@app.route("/set_uid/<uuid:id>")
def get_setid(id):
    print(id)
    return str(id)

@app.route("/any/<any(apple,xiaomi,huawei):bland>")
def get_any(bland):
    print(bland)
    return bland
#
# *******************************************************

if __name__ =="__main__":
    app.run(debug=True)