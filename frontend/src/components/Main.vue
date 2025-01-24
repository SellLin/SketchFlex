<template>
 <!-- The overall div-->
<div style="background-color: #001f3f;">
  <!-- overall row-->
  <el-row>
      <!-- empty col for layout left part-->
      <el-col :span="4"></el-col>

      <!-- Here draws the sketch and input regional prompts-->
      <el-col :span="8">
          <div id="TestPrompt" style="width:900px;height:940px;margin-left:0px;background-color: white;border-radius: 8px;margin-top:40px">
            <SketchPad/>

            <el-button @click="postSemanticInference()" style = "margin-top: 1%;margin-bottom: 5%;">
                Inference
            </el-button>

            <el-button @click="postMask()" style = "margin-top: 1%;margin-bottom: 5%; margin-left: 15%;">
                Generate
            </el-button>

            <el-button @click="cleanPrompt()">
                Clean
            </el-button>
            <br>
        </div>
      </el-col>

      <!-- The third panel stuff for shape anchor and generated result-->
      <el-col :span="4" style="margin-left: 102px;margin-top:280px">
          <div id="Disentangle" style="width:530px;height:650px;margin-left:0px;background-color: white;border-radius: 8px;margin-top:40px">
              <div class="header2" style="width:94%;margin-left: 3%;">
                  <p style="font-size: 18px;font-weight: bold;">Refine Sketch</p>
              </div>

              <el-button @click="genSingleObject()" style = "margin-top: 1%;margin-bottom: 1%; margin-left: -50px;">
                      Generate
              </el-button>

              <el-button @click="showGallery()">
                      Show
              </el-button>

              <el-button @click="addDragMask" style = "margin-top: 1%;margin-bottom: 1%; margin-left: 130px;">
                Show
              </el-button>
              <el-button @click="updateDragMask" style = "margin-top: 1%;margin-bottom: 1%;">
                Save
              </el-button>
            
              <!-- <div style="display: flex; align-items: flex-start; margin-left: 0;"> -->
                <div style="display: flex;">
                    <!-- First container -->
                    <div>
                        <!-- First corview div with custom scrollbar -->
                        <div id="corview1" style="width: 230px; height: 100px; margin-top: 5px;margin-left: 10px;overflow-y: auto;margin-bottom: 20px;">
 
                            <img v-for="(path, i) in gallery_list" @click="clickGallery" selected="unselected" 
                                style="display: block; margin-left: 5px; margin-top: 10px; border-radius: 0px;" 
                                width="90" height="90" class="galleryImg" 
                                :id="'galleryImg' + (i).toString()">
                        </div>

                        <!-- Second corview div with custom scrollbar -->
                        <div id="corview2" style="width: 230px; height: 100px; overflow-y: auto; margin-left: 10px;margin-top: 5px;margin-bottom: 20px;">
                            <img v-for="(path, i) in gallery_list2" @click="clickGallery" selected="unselected" 
                                style="display: block; margin-left: 5px; margin-top: 10px; border-radius: 0px;" 
                                width="90" height="90" class="galleryImg" 
                                :id="'galleryImga' + (i).toString()">
                        </div>
                    </div>

                    <!-- Canvas on the right -->
                    <canvas id="fabricCanvas" width="256" height="256" style="margin-left: 20px; z-index: 9998;"></canvas>
                </div>

              <div class="header2" style="width:94%;margin-left: 3%;">
                  <p style="font-size: 18px;font-weight: bold;">Result</p>
              </div>
                  <!-- Dropdown (Select) -->
                  <select v-model="selectedOption" style="margin-left: 430px;width:44px;height:14px; color: #000 !important">
                      <option value="1">1</option>
                      <option value="2">2</option>
                      <option value="3">3</option>
                  </select>
                  
                  <!-- Folder Button -->
                  <button style="background: none; border: none; cursor: pointer;">
                      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                          <path d="M10,4L12,6H20C21.1,6 22,6.9 22,8V18C22,19.1 21.1,20 20,20H4C2.9,20 2,19.1 2,18V6C2,4.9 2.9,4 4,4H10M4,8V18H20V8H4Z" />
                      </svg>
                  </button>
                  <canvas id="resultCanvas" width="256" height="256" style="margin-left: 20px; z-index: 9998;"></canvas>
                  
          </div>
      </el-col>
  </el-row>
  <!-- end of overall row-->


<!-- <div id="corr" style="display:inline-block">
</div> -->


<!-- The bottom few lines-->
<div id="cor_im" style="display:inline-block">
<img id="cor_image" width="150" height="350" style="visibility: hidden;">
</div>

</div>
</template>
<!-- <script src="https://cdnjs.cloudflare.com/ajax/libs/d3-lasso/0.6.0/d3-lasso.min.js"></script> -->
<script>
import 'd3-lasso'
import { ElMessage, ElMessageBox } from "element-plus";
import { useDataStore } from "../stores/counter";
import { postSketch, postGallery, postInference,postClean, postGenSingle,postAnchorSingle,postResult, postMaskUpdate } from "../service/module/dataService";
import SketchPad from "./SketchPad.vue";
import { markRaw } from 'vue';

export default {
    name: 'APP',
    props: ["msgH"],
    data() {
        return {
          selectedOption: 1,  
            msg1: "Hello, main!",
            formData: {
                name: '',
                region: '',
                price: '',
                rarity: '',
                collection: '',
                backgroundColor: '',
                style: '',
                type: '',
            },
            formData1: {
                name: '',

            },
            formData2: {
                name: '',

            },
            formDataA: {
                type: '',
                attr: '',
                state: '',
                direction: '',
                neg: '',
                relation: '',
            },
            gallery_list: [0,0,0,0,0,0,0,0],
            gallery_list2: [0,0,0,0,0,0,0,0],
            result_list: [0,0,0,0,0,0,0,0],
            border_color_1 : '',
            border_color_2 : '',
            result_img: '',
            choose_img: '',
            mask_canvas: null,
            result_canvas: null,
            mask_instance_num: 4,
            mask_transformation: [],

        }
    },
    components: {
    SketchPad,
    },
    
    methods: {
        //把mask数据以图片形式传到后端
        async postMask(){
            let formData = new FormData();
            const dataStore = useDataStore();
            formData.append("search_txt", this.formData.name);
            formData.append("sketch_draw", dataStore.sketchDrawBlob, "draw_img.png");
            formData.append("sketch_rect", dataStore.sketchRectBlob, "rect_img.png");
            formData.append("sketch_txt", dataStore.sketchTxtBlob, "txt_img.png");
            formData.append("rects_info", JSON.stringify(dataStore.sketchRectsInfo));
            formData.append("txts_info", JSON.stringify(dataStore.sketchTxtsInfo));
            formData.append("if_preview", 'result');

            const canvas = document.getElementById('fabricCanvas');
            const rCanvas = document.getElementById('resultCanvas');
            const ctx = canvas.getContext('2d');
            const rctx = rCanvas.getContext('2d');
            // ctx.clearRect(0,0,canvas.width,canvas.height);
            // 创建一个 Image 对象
            const img = new Image();

            try {
              // 先执行 postSketch，并等待其完成
              await new Promise((resolve, reject) => {
                postSketch(formData, (res) => {
                  console.log(res);
                  resolve(); // 表示 postSketch 完成
                });
              });

              // postSketch 完成后再执行 postResult
              postResult(formData, (res) => {
                console.log(res);
                let file_names = res["file_names"].split("----");
                for (let i = 0; i < file_names.length; i++) {
                  let timestamp = new Date().getTime();
                  d3.select("#resultImg" + i)
                    .attr("src", file_names[i] + "?t=" + timestamp)
                    .style("display", "inline-block");
                }
              });

              // 尝试更新canvas
                          // 设置图片的源路径
            img.src = './src/assets/generated_result/result.png'+ '?t=' + new Date().getTime(); // 替换为你的图片路径
            // 图片加载完成后，将其绘制到 canvas 上
            img.onload = function() {
                // 将图片绘制到 canvas 上，从坐标 (0, 0) 开始绘制
                // ctx.drawImage(img, 0, 0, 1024, 1024, 0, 0,512,512 );
                rctx.drawImage(img, 0, 0, 1024, 1024, 0, 0,200,200 );
            };
            } catch (error) {
              console.error("An error occurred:", error);
            }

        },
        postSemanticInference(){
            let formData = new FormData();
            const dataStore = useDataStore();
            formData.append("search_txt", this.formData.name);
            formData.append("sketch_draw", dataStore.sketchDrawBlob, "draw_img.png");
            formData.append("sketch_rect", dataStore.sketchRectBlob, "rect_img.png");
            formData.append("sketch_txt", dataStore.sketchTxtBlob, "txt_img.png");
            formData.append("rects_info", JSON.stringify(dataStore.sketchRectsInfo));
            formData.append("txts_info", JSON.stringify(dataStore.sketchTxtsInfo));
            console.log('posting mask')
            postInference(formData, (res) => {
            console.log(res);
            });
        },
        cleanPrompt(){
          let formData = new FormData();
            postClean(formData, (res) => {
            console.log(res);
            });
        },
        // 生成单object图片
        genSingleObject(){
            let formData = new FormData();
            const dataStore = useDataStore();
            formData.append("search_txt", this.formData.name);
            formData.append("sketch_draw", dataStore.sketchDrawBlob, "draw_img.png");
            formData.append("sketch_rect", dataStore.sketchRectBlob, "rect_img.png");
            formData.append("sketch_txt", dataStore.sketchTxtBlob, "txt_img.png");
            formData.append("rects_info", JSON.stringify(dataStore.sketchRectsInfo));
            formData.append("txts_info", JSON.stringify(dataStore.sketchTxtsInfo));
            formData.append("color_info", JSON.stringify(dataStore.color));
            postGenSingle(formData, (res) => {
                console.log('gen single ....')
            });
        },
        // 显示gallery 图片
        showGallery(){
            let formData = new FormData();
            const dataStore = useDataStore();
            formData.append("search_txt", this.formData.name);
            formData.append("sketch_draw", dataStore.sketchDrawBlob, "draw_img.png");
            formData.append("sketch_rect", dataStore.sketchRectBlob, "rect_img.png");
            formData.append("sketch_txt", dataStore.sketchTxtBlob, "txt_img.png");
            formData.append("rects_info", JSON.stringify(dataStore.sketchRectsInfo));
            formData.append("txts_info", JSON.stringify(dataStore.sketchTxtsInfo));
            formData.append("color_info", JSON.stringify(dataStore.color));
            postGallery(formData, (res) => {
            console.log(res);
            let file_names=res["file_names"].split("----")

            let galleryImg = d3.select("#galleryImg" + 1);
            let galleryImga = d3.select("#galleryImga" + 1);
            let galleryImgSrc = galleryImg.attr("src"); // Get the current src attribute of #galleryImg
            let galleryImgaSrc = galleryImga.attr("src"); // Get the current src attribute of #galleryImga
            if (!galleryImgSrc) { // If #galleryImg's src is empty
            this.border_color_1 = dataStore.color
            } else if (!galleryImgaSrc) { // If #galleryImg's src is not empty, but #galleryImga's src is empty
            this.border_color_2 = dataStore.color
          }else {
            this.border_color_1 = this.border_color_2
            this.border_color_2 = dataStore.color
          }

            for (let i = 0; i < file_names.length; i++) {
                let timestamp = new Date().getTime();
                let galleryImg = d3.select("#galleryImg" + i);
                let galleryImga = d3.select("#galleryImga" + i);
                
                let galleryImgSrc = galleryImg.attr("src"); // Get the current src attribute of #galleryImg
                let galleryImgaSrc = galleryImga.attr("src"); // Get the current src attribute of #galleryImga

                if (!galleryImgSrc) { // If #galleryImg's src is empty
                    galleryImg
                        .attr("src", file_names[i] + "?t=" + timestamp)
                        .style("display", "inline-block");
                } else if (!galleryImgaSrc) { // If #galleryImg's src is not empty, but #galleryImga's src is empty
                    galleryImga
                        .attr("src", file_names[i] + "?t=" + timestamp)
                        .style("display", "inline-block");
                } else { // If both #galleryImg and #galleryImga have non-empty src attributes
                    // Replace #galleryImg's content with #galleryImga's src
                    galleryImg
                        .attr("src", galleryImgaSrc);

                    // Update #galleryImga with the new file_names[i]
                    galleryImga
                        .attr("src", file_names[i] + "?t=" + timestamp)
                        .style("display", "inline-block");
                }
            }
              // 获取要修改的元素
              const corview1 = document.getElementById('corview1');
              const corview2 = document.getElementById('corview2');

              // 设置边框颜色
              corview1.style.border = `2px solid ${this.border_color_1}`;
              corview2.style.border = `2px solid ${this.border_color_2}`;

            });
        },
        // 点击选中gallery 图片
        clickGallery(event){
            d3.selectAll(".galleryImg").style("border", "none")
            event.target.style.border="solid orange 4px"
            console.log(event.target.src)
            this.choose_img = event.target.src
            let formData = new FormData();
            const dataStore = useDataStore();
            formData.append("search_txt", this.formData.name);
            formData.append("anchor_image_info", this.choose_img);
            formData.append("sketch_draw", dataStore.sketchDrawBlob, "draw_img.png");
            formData.append("sketch_rect", dataStore.sketchRectBlob, "rect_img.png");
            formData.append("sketch_txt", dataStore.sketchTxtBlob, "txt_img.png");
            formData.append("rects_info", JSON.stringify(dataStore.sketchRectsInfo));
            formData.append("txts_info", JSON.stringify(dataStore.sketchTxtsInfo));
            formData.append("color_info", JSON.stringify(dataStore.color));
            formData.append("if_preview", 'preview');
            postAnchorSingle(formData, (res) => {
                console.log('gen single ....')
            });
            
            postResult(formData, (res) => {
            console.log(res);
            let file_names=res["file_names"].split("----")
            for (let i=0;i<file_names.length; i++){
              let timestamp = new Date().getTime();
              d3.select("#resultImg"+i)
                  .attr("src", file_names[i] + "?t=" + timestamp)
                  .style("display", "inline-block");
                }
            });
            this.addDragMask()
        },
        updateData() {
          console.log('this.formData');
          const dataStore = useDataStore();
          this.formDataA.type = dataStore.type;
          this.formDataA.type = dataStore.type;
          this.formDataA.attr = dataStore.attribute;
          this.formDataA.state = dataStore.state;
          this.formDataA.direction = dataStore.direction;
          this.formDataA.neg = dataStore.negative;
        },
        //测试拖动mask后update
        updateDragMask(){
          let formData = new FormData();
          const dataURL = this.mask_canvas.toDataURL({
              format: 'png',
              quality: 1.0
          });

          // Convert the data URL to a Blob
          function dataURLToBlob(dataURL) {
              const byteString = atob(dataURL.split(',')[1]);
              const mimeString = dataURL.split(',')[0].split(':')[1].split(';')[0];
              const ab = new ArrayBuffer(byteString.length);
              const ia = new Uint8Array(ab);
              for (let i = 0; i < byteString.length; i++) {
                  ia[i] = byteString.charCodeAt(i);
              }
              return new Blob([ab], { type: mimeString });
          }

          const imageBlob = dataURLToBlob(dataURL);
          formData.append('mask_image', imageBlob, 'mask_canvas_image.png');
          formData.append('num', this.mask_instance_num);
          for (let i=0; i<this.mask_instance_num;i++){
            // formData.append('left'+i, this.mask_transformation[i]["left"]);
            // formData.append('top'+i, this.mask_transformation[i]["top"]);
            formData.append('left_move'+i, this.mask_transformation[i]["left"]-this.mask_transformation[i]["initial_left"]);
            console.log('left_move:',this.mask_transformation[i]["left"]-this.mask_transformation[i]["initial_left"])
            formData.append('top_move'+i, this.mask_transformation[i]["top"]-this.mask_transformation[i]["initial_top"]);
            formData.append('scaleX'+i, this.mask_transformation[i]["scaleX"]);
            formData.append('scaleY'+i, this.mask_transformation[i]["scaleY"]);
            formData.append('modified'+i, this.mask_transformation[i]["modified"]);
          }

          postMaskUpdate(formData, (res) => {
            console.log(res);
            });
        },
        //测试添加可拖动mask
        addDragMask(){
          const canvass = document.getElementById('fabricCanvas');
          const ctx = canvass.getContext('2d');
          ctx.clearRect(0,0,canvass.width,canvass.height);

          let canvas=this.mask_canvas
          canvas.clear();
          console.log(canvas)

          let makeWhiteTransparent = (image) => {
            image.filters.push(new fabric.Image.filters.RemoveColor({
              color: '#FFFFFF',  // White color to be removed
              distance: 0.1      // Tolerance for color matching
            }));
            image.applyFilters();
          };

          function getBoundingBoxForImage(imgElement) {
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            
            canvas.width = imgElement.width;
            canvas.height = imgElement.height;
            context.drawImage(imgElement, 0, 0, canvas.width, canvas.height);
            
            const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageData.data;

            let minX = canvas.width, minY = canvas.height, maxX = 0, maxY = 0;

            for (let y = 0; y < canvas.height; y++) {
              for (let x = 0; x < canvas.width; x++) {
                // const alpha = data[(y * canvas.width + x) * 4 + 3]; // Alpha channel
                let index = (y * canvas.width + x) * 4;
                let r = data[index];     // Red channel
                let g = data[index + 1]; // Green channel
                let b = data[index + 2]; // Blue channel
                let alpha = data[index + 3]; // Alpha channel
                if (!(r === 255 && g === 255 && b === 255) && alpha > 0)
                { // Check if pixel is not transparent
                  minX = Math.min(minX, x);
                  minY = Math.min(minY, y);
                  maxX = Math.max(maxX, x);
                  maxY = Math.max(maxY, y);
                }
              }
            }

            return {
              x: minX,
              y: minY,
              width: maxX - minX + 1,
              height: maxY - minY + 1
            };
          }

          let instances=[]
          let color = ['red','green','yellow','blue']
          this.mask_transformation=[]

          for (let i=this.mask_instance_num-1; i>=0;i--){
            instances.push(color[i] + "/anchor_img_mask_output_with_color.png")
            this.mask_transformation.push({'file_name':color[i] + "/anchor_img_mask_output_with_color.png", "scaleX": 1, "scaleY": 1, "left": 0, "top": 0, "modified": "false", "initial_left": 0, "initial_top": 0})

            let imgElement = new Image();
            imgElement.src = '../src/assets/anchor_images/' + color[i] + "/anchor_img_mask_output_with_color.png";
            imgElement.onerror = function() {
                console.log('Image not found: ' + imgElement.src + ', skipping...');
                instances.splice(i, 1); 
                this.mask_transformation.splice(i, 1); 
            };
          }

          let _this=this

          // instances.forEach((instance, i) => {
          //   let imgElement = new Image();
          //   imgElement.src = '../src/assets/anchor_images/' + instance + '?t=' + new Date().getTime();
          //   imgElement.onload = function() {
          //     let bbox = getBoundingBoxForImage(imgElement);
          //     fabric.Image.fromURL('../src/assets/anchor_images/' + instance + '?t=' + new Date().getTime(), function(img) {
          //       makeWhiteTransparent(img)

          //       img.set({
          //         selectable: true,
          //         perPixelTargetFind: true,  // Enable per-pixel selection
          //         targetFindTolerance: 4,    // Adjust tolerance for pixel targeting
          //         hasControls: true,  // Enable control handles for scaling and rotating
          //         cornerSize: 15,     // Size of the corner controls
          //         transparentCorners: false,  // Make corner controls more visible
          //         left: bbox.x,
          //         top: bbox.y,
          //         width: bbox.width,
          //         height: bbox.height,
          //         cropX: bbox.x,
          //         cropY: bbox.y,
          //       });
          //       _this.mask_transformation[i]["left"]=bbox.x
          //       _this.mask_transformation[i]["top"]=bbox.y
          //       _this.mask_transformation[i]["initial_left"]=bbox.x
          //       _this.mask_transformation[i]["initial_top"]=bbox.y
          //       canvas.add(markRaw(img));

          //       img.on('modified', function() {
          //           const scaleX = img.scaleX;
          //           const scaleY = img.scaleY;
          //           const left = img.left;
          //           const top = img.top;
          //           _this.mask_transformation[i]["left"]=left
          //           _this.mask_transformation[i]["top"]=top
          //           _this.mask_transformation[i]["scaleX"]=scaleX
          //           _this.mask_transformation[i]["scaleY"]=scaleY
          //           _this.mask_transformation[i]["modified"]="true"
          //           console.log('Scaling and Translation recorded:');
          //           console.log('ScaleX:', scaleX, 'ScaleY:', scaleY);
          //           console.log('Left:', left, 'Top:', top);
          //       });
          //     });
          // }
          // });

          instances.forEach((instance, i) => {
        let imgElement = new Image();
        imgElement.src = '../src/assets/anchor_images/' + instance + '?t=' + new Date().getTime();
        imgElement.onload = function() {
            // Resize the image to 256x256
            let tempCanvas = document.createElement('canvas');
            tempCanvas.width = 256;
            tempCanvas.height = 256;
            let tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(imgElement, 0, 0, 256, 256);

            // Create a new image element with the resized image
            let resizedImgElement = new Image();
            resizedImgElement.src = tempCanvas.toDataURL();

            resizedImgElement.onload = function() {
                let bbox = getBoundingBoxForImage(resizedImgElement);
                fabric.Image.fromURL(resizedImgElement.src, function(img) {
                    makeWhiteTransparent(img);

                    img.set({
                        selectable: true,
                        perPixelTargetFind: true,  // Enable per-pixel selection
                        targetFindTolerance: 4,    // Adjust tolerance for pixel targeting
                        hasControls: true,  // Enable control handles for scaling and rotating
                        cornerSize: 15,     // Size of the corner controls
                        transparentCorners: false,  // Make corner controls more visible
                        left: bbox.x,
                        top: bbox.y,
                        width: bbox.width,
                        height: bbox.height,
                        cropX: bbox.x,
                        cropY: bbox.y,
                    });
                    _this.mask_transformation[i]["left"] = bbox.x;
                    _this.mask_transformation[i]["top"] = bbox.y;
                    _this.mask_transformation[i]["initial_left"] = bbox.x;
                    _this.mask_transformation[i]["initial_top"] = bbox.y;
                    canvas.add(markRaw(img));

                    img.on('modified', function() {
                        const scaleX = img.scaleX;
                        const scaleY = img.scaleY;
                        const left = img.left;
                        const top = img.top;
                        _this.mask_transformation[i]["left"] = left;
                        _this.mask_transformation[i]["top"] = top;
                        _this.mask_transformation[i]["scaleX"] = scaleX;
                        _this.mask_transformation[i]["scaleY"] = scaleY;
                        _this.mask_transformation[i]["modified"] = "true";
                        console.log('Scaling and Translation recorded:');
                        console.log('ScaleX:', scaleX, 'ScaleY:', scaleY);
                        console.log('Left:', left, 'Top:', top);
                    });
                });
            };
        };
    });
        },
    },
    created() {
        // this.drawCor();
        // for (let i=0;i<10;i++)
        // {
        //     this.conceptList.push('')
        // }

        // this.conceptGroup = this.conceptOptions.map(option => option.value);

    },
    mounted() {
        // this.drawCor();
        this.canvas1= new fabric.Canvas('c1')
        this.mask_canvas = new fabric.Canvas('fabricCanvas');
        this.result_canvas = new fabric.Canvas('resultCanvas');
        // this.conceptGroup = this.conceptOptions.map(option => option.value);
    },
    // created() {
    // },
    // mounted() {
    //     // this.$refs.img_button.click(() => {
    //     //     this.$refs.img_sub.change();
    //     // })
    // },

    watch: {
        // formData() {
        //     console.log(this.formData);
        // }
    }
}

</script>


<style>
*::-webkit-scrollbar {
  width: 3px !important;
  min-width: 3px !important;
}

/* 滚动条轨道 */
*::-webkit-scrollbar-track {
  background: transparent;
}

/* 滚动条滑块 */
*::-webkit-scrollbar-thumb {
  background-color: #ca4343; /* 滑块颜色 */
  border-radius: 3px;
}

/* 为 Firefox 定义滚动条样式 */
* {
  scrollbar-width: thin; /* 滚动条宽度 */
  scrollbar-color: #c5c5c5 transparent; /* 滑块颜色和轨道背景色 */
  margin: 0;
  padding: 0;
}

/* .headline_img{
  width: 1550px;
  height: 350px;
  background: url('E:/CHI2025/repo/RegionDrawing/InterfaceFrontend/public/city.png');
} */
.headline_img {
  width: 1550px;
  height: 350px;
  background-image: url('E:/CHI2025/repo/RegionDrawing/InterfaceFrontend/public/city.png'); /* 使用两倍分辨率的图片 */
  background-size: 1550px 350px; /* 强制背景图片缩放到指定大小 */
  background-repeat: no-repeat; /* 防止图片重复 */
  background-position: center center; /* 图片居中显示 */
}


/* .el-input {
    --el-input-border-radius: 25px;
    height: 30px;
    width:400px;
    font-size: 15px;
    font-family: Quicksand, system-ui;
    color: black
      
} */

/* .el-form-item__content {
    background-color: #0000;
} */
.axis {
  stroke: #000;
  stroke-width: 1.5px;
}

.node circle {
  stroke: #000;
}

.link {
  /* fill: none; */
  fill: blue;
  stroke: #999;
  stroke-width: 1.5px;
  stroke-opacity: .3;
}
/* .path {
    fill: blue
} */
.link.active {
  stroke: red;
  stroke-width: 2px;
  stroke-opacity: 1;
}

.node circle.active {
  stroke: red;
  stroke-width: 3px;
}

.tooltip{
    z-index:1;
    width:300px;
    height: 300px;
    border: 1px solid;
    position: absolute;
    /* position:relative; */
    visibility: hidden;
    background-color: white;
}

.avatar-uploader .el-upload {
    border: 1px dashed #d9d9d9;
    border-radius: 6px;
    cursor: pointer;
    position: relative;
    overflow: hidden;
  }
  .avatar-uploader .el-upload:hover {
    border-color: #409EFF;
  }
  .avatar-uploader-icon {
    font-size: 28px;
    color: #8c939d;
    width: 178px;
    height: 178px;
    line-height: 178px;
    text-align: center;
  }
  .avatar {
    width: 178px;
    height: 178px;
    display: block;
  }

/* .lasso path {
stroke: #2378ae;
stroke-width: 3;

}

.lasso #drawn {
fill-opacity: 0.05;
}

.lasso #loop_close {

stroke: #2378ae;
stroke-width: 3;
}

.lasso #origin {
fill: rgb(180, 180, 180);
fill-opacity: 0.5;
} */
.contour {
  mix-blend-mode: multiply;
}

.header {
    text-align: center;
    position: relative;
    /* padding: 20px 0; */
    padding: 2px 0;
  }
  .header::before,
  .header::after {
    content: "";
    position: absolute;
    top: 50%;
    /* top: 20%; */
    width: 40%;
    border-top: 1px dashed #000;
  }
  .header::before {
    left: 0;
  }
  .header::after {
    right: 0;
  }

  .header2 {
    text-align: center;
    position: relative;
    /* padding: 20px 0; */
    padding: 2px 0;
  }
  .header2::before,
  .header2::after {
    content: "";
    position: absolute;
    top: 50%;
    /* top: 20%; */
    width: 25%;
    border-top: 1px dashed #000;
  }
  .header2::before {
    left: 0;
  }
  .header2::after {
    right: 0;
  }

  html {
  font-family: Verdana, Geneva, Tahoma, sans-serif;
    }
    *::-webkit-scrollbar {
    width: 10px;
    }

    *::-webkit-scrollbar-thumb {
    background: #ccc;
    border-radius: 5px;
    }

    * {
    scrollbar-width: 10px;
    scrollbar-base-color: green;
    scrollbar-track-color: red;
    scrollbar-arrow-color: blue;
    }

    .text-element {
    cursor: default; /* or any cursor value you prefer, e.g., auto, inherit, etc. */
}

/*Chrome*/
@media screen and (-webkit-min-device-pixel-ratio:0) {
    input[type='range'] {
      overflow: hidden;
      width: 150px;
      -webkit-appearance: none;
      /* background-color: #9a905d; */
      background-color: rgb(230, 230, 230);

    }
    
    input[type='range']::-webkit-slider-runnable-track {
      height: 10px;
      -webkit-appearance: none;
      color: #919191;
      margin-top: -1px;
    }
    
    input[type='range']::-webkit-slider-thumb {
      width: 10px;
      -webkit-appearance: none;
      height: 10px;
      cursor: ew-resize;
      background: #434343;
      box-shadow: -80px 0 0 80px #919191;
    }

}
/** FF*/
/* input[type="range"]::-moz-range-progress {
  background-color: #43e5f7; 
}
input[type="range"]::-moz-range-track {  
  background-color: #9a905d;
} */
/* IE*/
/* input[type="range"]::-ms-fill-lower {
  background-color: #43e5f7; 
}
input[type="range"]::-ms-fill-upper {  
  background-color: #9a905d;
} */

.custom-button{
  background-color: "yellow";
  border-color: "yellow";
  color: "white";
}

/* Define hover and active states */
.custom-button:hover {
  background-color: "yellow";
  border-color: "yellow"
}

.custom-button:active {
  background-color: "yellow";
  border-color: "yellow"
}
</style>
