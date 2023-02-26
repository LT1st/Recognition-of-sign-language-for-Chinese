# 加载Flask框架主模块和render_template模板渲染模块
from flask import Flask, render_template, Response, jsonify

# 创建Flask服务
import demo

app = Flask(__name__)

# 访问URL：http://127.0.0.1:8080
# 返回网页index.html
@app.route('/')
def index():
    return render_template('test.html')

# 访问URL：http://127.0.0.1:8080/home/hello
# 返回结果：{"data":"welcome to use flask.","msg":"hello"}
@app.route('/RSL/', methods=["POST"])
def RSL():
    print("sssss")
    aa=demo.run()
    message_json = {
        "message": aa
    }
    return jsonify(message_json)

if __name__ == "__main__":
    # 启动Flask服务，指定主机IP和端口
    app.run(host='127.0.0.1', port=8080)
