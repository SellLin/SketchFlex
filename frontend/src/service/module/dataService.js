import axios from 'axios';

// axios.defaults.withCredentials = true
// const TEST_URL_PREFIX = 'http://nftsea.natapp1.cc/api/test';
const TEST_URL_PREFIX = 'http://127.0.0.1:5000/api/test';
const LORA_URL_PREFIX = 'http://10.4.128.164:5000';
// const LORA_URL_PREFIX = 'http://192.168.224.1:5000';

export function fetchHello(param, callback) {
    const url = `${TEST_URL_PREFIX}/hello/`;
    axios.post(url, param)
        .then(response => {
            callback(response.data)
        }, errResponse => {
            console.log(errResponse)
        })
}

// 发送草稿数据给生成
export function postSketch(param, callback) {
    const url = `${TEST_URL_PREFIX}/getSketch/`;
    axios
      .post(url, param, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      .then(
        (response) => {
          callback(response.data);
        },
        (errResponse) => {
          console.log(errResponse);
        }
      );
  }

// 发送草稿数据给语义推导
export function postInference(param, callback) {
  const url = `${TEST_URL_PREFIX}/getInference/`;
  axios
    .post(url, param, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    })
    .then(
      (response) => {
        callback(response.data);
      },
      (errResponse) => {
        console.log(errResponse);
      }
    );
}

// 清除所有语义
export function postClean(param, callback) {
  const url = `${TEST_URL_PREFIX}/getClean/`;
  axios
    .post(url, param, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    })
    .then(
      (response) => {
        callback(response.data);
      },
      (errResponse) => {
        console.log(errResponse);
      }
    );
}

export function postCleanMask(param, callback) {
  const url = `${TEST_URL_PREFIX}/getCleanMask/`;
  axios
    .post(url, param, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    })
    .then(
      (response) => {
        callback(response.data);
      },
      (errResponse) => {
        console.log(errResponse);
      }
    );
}

// 单 object 生成
export function postGenSingle(param, callback) {
  const url = `${TEST_URL_PREFIX}/getGenSingle/`;
  axios
    .post(url, param, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    })
    .then(
      (response) => {
        callback(response.data);
      },
      (errResponse) => {
        console.log(errResponse);
      }
    );
}

// 读取gallery数据
export function postGallery(param, callback) {
  const url = `${TEST_URL_PREFIX}/getGallery/`;
  axios
    .post(url, param,{
      headers: {
        "Content-Type": "multipart/form-data",
      },
    })
    .then(
      (response) => {
      callback(response.data);
      },
      (errResponse) => {
      console.log(errResponse);
      }
  );
}

// 提交选择的anchor image数据
export function postAnchorSingle(param, callback) {
  const url = `${TEST_URL_PREFIX}/getAnchorSingle/`;
  axios
    .post(url, param,{
      headers: {
        "Content-Type": "multipart/form-data",
      },
    })
    .then(
      (response) => {
      callback(response.data);
      },
      (errResponse) => {
      console.log(errResponse);
      }
  );
}

// 发送mask拖拽数据
export function postMaskUpdate(param, callback) {
  const url = `${TEST_URL_PREFIX}/getMaskUpdate/`;
  axios
    .post(url, param, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    })
    .then(
      (response) => {
        callback(response.data);
      },
      (errResponse) => {
        console.log(errResponse);
      }
    );
}


// 发送json数据
  export function postJson(param, callback) {
    const url = `${TEST_URL_PREFIX}/getJson/`;
    axios.post(url, param).then(
        (response) => {
        callback(response.data);
        },
        (errResponse) => {
        console.log(errResponse);
        }
    );
  }

// 获取本地语义json
export function postUpdateJson(param, callback) {
  const url = `${TEST_URL_PREFIX}/updateJson/`;
  axios.post(url, param).then(
      (response) => {
      callback(response.data);
      },
      (errResponse) => {
      console.log(errResponse);
      }
  );
}

// 获取生成的图像或预览mask图
export function postResult(param, callback) {
  const url = `${TEST_URL_PREFIX}/getResult/`;
  axios
    .post(url, param,{
      headers: {
        "Content-Type": "multipart/form-data",
      },
    })
    .then(
      (response) => {
      callback(response.data);
      },
      (errResponse) => {
      console.log(errResponse);
      }
  );
}




  export function postText(param, callback) {
    const url = `${TEST_URL_PREFIX}/getText/`;
    axios.post(url, param).then(
      (response) => {
        callback(response.data);
      },
      (errResponse) => {
        console.log(errResponse);
      }
    );
  }

  export function postImg(param, callback) {
    const url = `${TEST_URL_PREFIX}/getImg/`;
    console.log("img post: ", param);
    axios.post(url, param).then(
      (response) => {
        callback(response.data);
      },
      (errResponse) => {
        console.log(errResponse);
      }
    );
  }

  export function fetchValue(param, callback) {
    const url = `${TEST_URL_PREFIX}/getValue/`;
    axios.get(url, param).then(
      (response) => {
        callback(response.data);
      },
      (errResponse) => {
        console.log(errResponse);
      }
    );
  }










