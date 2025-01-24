<template>
  <div style="display:inline-block; width: 540px; height:630px; margin-left: -170px;margin-top: 10px; border: ridge 1px;">
  <div class="sketch_pad_box" ref="sketch_pad_all_box" style="display:inline-block;">
    <canvas ref="sketch_bg_canvas" class="sketch_bg_canvas"></canvas>
    <canvas ref="sketch_bg_canvas2" class="sketch_bg_canvas"></canvas>
    <canvas ref="sketch_canvas" @click="handle_canvas_sel_click"></canvas>
    <div
      class="sel_rect_box"
      :style="{
        width: now_select_rect_pos.w + 'px',
        height: now_select_rect_pos.h + 'px',
        top: now_select_rect_pos.y + 'px',
        left: now_select_rect_pos.x + 'px',
      }"
      v-show="now_selecting_rect"
      @mousedown="start_move_rect"
    >
      <div class="sel_e" @mousedown="start_resize_rect_size($event, 'e')"></div>
      <div class="sel_s" @mousedown="start_resize_rect_size($event, 's')"></div>
      <div class="sel_w" @mousedown="start_resize_rect_size($event, 'w')"></div>
      <div class="sel_n" @mousedown="start_resize_rect_size($event, 'n')"></div>
      <div
        class="sel_es"
        @mousedown="start_resize_rect_size($event, 'es')"
      ></div>
      <div
        class="sel_en"
        @mousedown="start_resize_rect_size($event, 'en')"
      ></div>
      <div
        class="sel_ws"
        @mousedown="start_resize_rect_size($event, 'ws')"
      ></div>
      <div
        class="sel_wn"
        @mousedown="start_resize_rect_size($event, 'wn')"
      ></div>
    </div>
    <div
      class="txt_edit_btn"
      :style="{
        left: `${now_select_txt_pos.x - 10}px`,
        top: `${now_select_txt_pos.y - 10}px`,
      }"
      v-show="now_selecting_txt"
      @mousedown="start_move_txt"
    >
      <svg
        v-show="edit_txt_status == 0"
        xmlns="http://www.w3.org/2000/svg"
        id="arrow-circle-down"
        viewBox="0 0 24 24"
        width="512"
        height="512"
      >
        <path
          d="M23.351,10.253c-.233-.263-.462-.513-.619-.67L20.487,7.3a1,1,0,0,0-1.426,1.4l2.251,2.29L21.32,11H13V2.745l2.233,2.194a1,1,0,0,0,1.4-1.426l-2.279-2.24c-.163-.163-.413-.391-.674-.623A2.575,2.575,0,0,0,12.028.006.28.28,0,0,0,12,0l-.011,0a2.584,2.584,0,0,0-1.736.647c-.263.233-.513.462-.67.619L7.3,3.513A1,1,0,1,0,8.7,4.939l2.29-2.251L11,2.68V11H2.68l.015-.015L4.939,8.7A1,1,0,1,0,3.513,7.3L1.274,9.577c-.163.163-.392.413-.624.675A2.581,2.581,0,0,0,0,11.989L0,12c0,.01.005.019.006.029A2.573,2.573,0,0,0,.65,13.682c.233.262.461.512.618.67l2.245,2.284a1,1,0,0,0,1.426-1.4L2.744,13H11v8.32l-.015-.014L8.7,19.062a1,1,0,1,0-1.4,1.425l2.278,2.239c.163.163.413.391.675.624a2.587,2.587,0,0,0,3.43,0c.262-.233.511-.46.669-.619l2.284-2.244a1,1,0,1,0-1.4-1.425L13,21.256V13h8.256l-2.2,2.233a1,1,0,1,0,1.426,1.4l2.239-2.279c.163-.163.391-.413.624-.675A2.589,2.589,0,0,0,23.351,10.253Z"
        />
      </svg>
      <svg
        xmlns="http://www.w3.org/2000/svg"
        xmlns:xlink="http://www.w3.org/1999/xlink"
        v-show="edit_txt_status == 1"
        version="1.1"
        id="Capa_1"
        x="0px"
        y="0px"
        viewBox="0 0 507.506 507.506"
        style="enable-background: new 0 0 507.506 507.506"
        xml:space="preserve"
        width="512"
        height="512"
      >
        <g>
          <path
            d="M163.865,436.934c-14.406,0.006-28.222-5.72-38.4-15.915L9.369,304.966c-12.492-12.496-12.492-32.752,0-45.248l0,0   c12.496-12.492,32.752-12.492,45.248,0l109.248,109.248L452.889,79.942c12.496-12.492,32.752-12.492,45.248,0l0,0   c12.492,12.496,12.492,32.752,0,45.248L202.265,421.019C192.087,431.214,178.271,436.94,163.865,436.934z"
          />
        </g>
      </svg>
    </div>
    <textarea
      name=""
      id="text_input_assist"
      ref="text_input_assist"
      v-show="now_adding_txt"
      v-model="txt_input_content"
      @blur="stay_txt_added"
      :style="{
        left: this.txt_input_pos.x + 'px',
        top: this.txt_input_pos.y + 'px',
        fontFamily: txt_input_font_family,
        fontSize: `${txt_input_size}px`,
      }"
      rows="1"
      cols="100"
    ></textarea>
    <div></div>
    <div class="sketch_btns_box">
      <div class="btn_box">
        <button
          @click="sel_sketch_tool(0)"
          :class="{ select_btn: now_sel_tool_idx == 0 }"
        >
          <svg
            id="Layer_1"
            height="512"
            viewBox="0 0 24 24"
            width="512"
            xmlns="http://www.w3.org/2000/svg"
            data-name="Layer 1"
          >
            <path
              d="m13.278 23.979-4.2-8.24-5.078 4.477v-18.197a2 2 0 0 1 3.212-1.591l13.905 12.008-6.617.734 4.145 8.13zm-3.594-11.445 4.474 8.766 1.789-.894-4.547-8.906 4.938-.547-10.386-8.973.042 13.81z"
            />
          </svg>
        </button>
      </div>
      <div class="btn_box">
        <button
          @click="sel_sketch_tool(1)"
          :class="{ select_btn: now_sel_tool_idx == 1 }"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            id="Layer_1"
            data-name="Layer 1"
            viewBox="0 0 24 24"
          >
            <!-- <path d="M2,19 L5,22 L19,8 L16,5 Z" fill="yellow" /> -->
            <path d="M20,1 L23,4 L20,7 L17,4 Z" fill="black" />
            <path
              d="M24,3.46c-.05-1.03-.54-1.99-1.34-2.64-1.43-1.17-3.61-1.01-4.98,.36l-1.67,1.67c-.81-.54-1.77-.84-2.77-.84-1.34,0-2.59,.52-3.54,1.46l-3.03,3.03c-.39,.39-.39,1.02,0,1.41s1.02,.39,1.41,0l3.03-3.03c.89-.89,2.3-1.08,3.42-.57L2.07,16.79c-.69,.69-1.07,1.6-1.07,2.57,0,.63,.16,1.23,.46,1.77l-1.16,1.16c-.39,.39-.39,1.02,0,1.41,.2,.2,.45,.29,.71,.29s.51-.1,.71-.29l1.16-1.16c.53,.3,1.14,.46,1.77,.46,.97,0,1.89-.38,2.57-1.07L22.93,6.21c.73-.73,1.11-1.73,1.06-2.76ZM5.8,20.52c-.62,.62-1.7,.62-2.32,0-.31-.31-.48-.72-.48-1.16s.17-.85,.48-1.16L16.08,5.61l2.32,2.32L5.8,20.52ZM21.52,4.8l-1.71,1.71-2.32-2.32,1.6-1.6c.37-.37,.85-.56,1.32-.56,.36,0,.7,.11,.98,.34,.37,.3,.58,.72,.61,1.19,.02,.46-.15,.92-.48,1.24Z"
              fill="black"
            />
          </svg>
        </button>
      </div>
      <div class="btn_box">
        <button
          @click="sel_sketch_tool(2)"
          :class="{ select_btn: now_sel_tool_idx == 2 }"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            id="Layer_1"
            data-name="Layer 1"
            viewBox="0 0 24 24"
          >
            <!-- <path d="M3,16 L8,21 L22.5,6.5 L17.5,1.5 Z" fill="yellow" /> -->
            <path
              d="M23.004,3.523l-2.526-2.527c-1.273-1.274-3.323-1.333-4.669-.132L2,13.178v6.701L-.061,21.939l2.121,2.121,2.061-2.061h6.701l12.314-13.809c1.2-1.344,1.142-3.395-.133-4.668Zm-2.105,2.671l-11.117,12.466-4.442-4.442L17.806,3.103h0c.159-.142,.4-.135,.55,.015l2.526,2.527c.15,.15,.157,.391,.016,.549Z"
            />
          </svg>
        </button>
      </div>
      <div class="btn_box">
        <button
          @click="sel_sketch_tool(3)"
          :class="{ select_btn: now_sel_tool_idx == 3 }"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            id="Outline"
            viewBox="0 0 24 24"
          >
            <path
              d="M15,7 L1,7 L1,21 L20,21 L20,12"
              stroke-width="1.8"
              stroke="#364856"
              fill="none"
              stroke-dasharray="3.5,2"
            />
            <path
              d="M20,3 L20,11"
              stroke-width="2.2"
              stroke="black"
              fill="none"
            />
            <path
              d="M16,7 L24,7"
              stroke-width="2.2"
              stroke="black"
              fill="none"
            />
          </svg>
        </button>
      </div>
      <div class="btn_box">
        <button
          @click="sel_sketch_tool(4)"
          :class="{ select_btn: now_sel_tool_idx == 4 }"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            id="Layer_1"
            data-name="Layer 1"
            viewBox="0 0 24 24"
          >
            <path
              d="m16,9c0,.553-.447,1-1,1h-2v6c0,.553-.447,1-1,1s-1-.447-1-1v-6h-2c-.553,0-1-.447-1-1s.447-1,1-1h6c.553,0,1,.447,1,1Zm8,11.5c0,1.93-1.57,3.5-3.5,3.5-1.393,0-2.599-.819-3.162-2H6.662c-.563,1.181-1.769,2-3.162,2-1.93,0-3.5-1.57-3.5-3.5,0-1.393.819-2.599,2-3.162V6.662c-1.181-.563-2-1.769-2-3.162C0,1.57,1.57,0,3.5,0c1.393,0,2.599.819,3.162,2h10.677c.563-1.181,1.769-2,3.162-2,1.93,0,3.5,1.57,3.5,3.5,0,1.393-.819,2.599-2,3.162v10.677c1.181.563,2,1.769,2,3.162Zm-4-3.464V6.964c-1.53-.22-2.744-1.434-2.964-2.964H6.964c-.22,1.53-1.434,2.744-2.964,2.964v10.072c1.53.22,2.744,1.434,2.964,2.964h10.072c.22-1.53,1.434-2.744,2.964-2.964Zm-1-13.536c0,.827.673,1.5,1.5,1.5s1.5-.673,1.5-1.5-.673-1.5-1.5-1.5-1.5.673-1.5,1.5ZM2,3.5c0,.827.673,1.5,1.5,1.5s1.5-.673,1.5-1.5-.673-1.5-1.5-1.5-1.5.673-1.5,1.5Zm3,17c0-.827-.673-1.5-1.5-1.5s-1.5.673-1.5,1.5.673,1.5,1.5,1.5,1.5-.673,1.5-1.5Zm17,0c0-.827-.673-1.5-1.5-1.5s-1.5.673-1.5,1.5.673,1.5,1.5,1.5,1.5-.673,1.5-1.5Z"
            />
          </svg>
        </button>
      </div>

      <div class="progress_box">
        <el-slider
          vertical
          v-model="lineWidth"
          :min="1"
          :max="100"
          :step="1"
        ></el-slider>
      </div>

      <div class="btn_box">
        <button
          @click="sel_sketch_tool(5)"
          :class="{ select_btn: now_sel_tool_idx == 5 }"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            id="Layer_1"
            data-name="Layer 1"
            viewBox="0 0 24 24"
          >
            <path
              d="m14.353,21l8.716-8.746c1.206-1.21,1.206-3.179,0-4.389l-5.935-5.955c-1.17-1.175-3.213-1.175-4.383,0L.882,13.82c-1.206,1.21-1.206,3.179,0,4.389l4.774,4.791h18.344v-2h-9.647Zm-.186-17.677c.416-.416,1.135-.416,1.551,0l5.935,5.955c.43.431.43,1.134,0,1.565l-5.504,5.523-7.49-7.515,5.509-5.527Zm-7.681,17.677l-4.188-4.203c-.43-.431-.43-1.134,0-1.565l4.949-4.966,7.49,7.515-3.207,3.218h-5.043Z"
            />
          </svg>
        </button>
      </div>

      <div class="btn_box">
        <button :class="{ off_btn: !backFlag }" @click="turn_back">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <g id="_01_align_center" data-name="01 align center">
              <path
                d="M24,24H22a7.008,7.008,0,0,0-7-7H10.17v6.414L.877,14.121a3,3,0,0,1,0-4.242L10.17.586V7H15a9.01,9.01,0,0,1,9,9ZM8.17,5.414,2.291,11.293a1,1,0,0,0,0,1.414L8.17,18.586V15H15a8.989,8.989,0,0,1,7,3.349V16a7.008,7.008,0,0,0-7-7H8.17Z"
              />
            </g>
          </svg>
        </button>
      </div>
      <!-- <div class="btn_box">
        <button>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            id="Outline"
            viewBox="0 0 24 24"
          >
            <path
              d="M0,23V16A9.01,9.01,0,0,1,9,7h4.83V5.414A2,2,0,0,1,17.244,4l5.88,5.879a3,3,0,0,1,0,4.242L17.244,20a2,2,0,0,1-3.414-1.414V17H8a6.006,6.006,0,0,0-6,6,1,1,0,0,1-2,0ZM15.83,8a1,1,0,0,1-1,1H9a7.008,7.008,0,0,0-7,7v1.714A7.984,7.984,0,0,1,8,15h6.83a1,1,0,0,1,1,1v2.586l5.879-5.879a1,1,0,0,0,0-1.414L15.83,5.414Z"
            />
          </svg>
        </button>
      </div> -->

    </div>

  </div>

  </div>
  <div id="palette" style="display:inline-block; margin-left: 620px;margin-top: -1100px">
    <br>
      <div class="label-row" v-for="(label, index) in labels" :key="index">
        <!-- Small color rectangle -->
        <div class="color-rect" :style="{ backgroundColor: label.color}" @click="changeColor"></div>
        <!-- Text entry -->
        <el-input v-model="label.text" placeholder="Enter text" style="width:100px"></el-input>
      </div>
      <div style="margin-top:25px;width: 300px;margin-left:-85px;">
                <p style="margin-left:10px;user-select: none;">Type:</p>
                <el-form :model="formDataA">
                    <el-input v-model="formDataA.type" placeholder="Please enter"  spellcheck="false" type="textarea" style="width:160px;margin-left: 5px;" :autosize="{ minRows: 1, maxRows: 2}">

                    </el-input>

                </el-form>
                <p style="margin-left:10px;user-select: none;">Attribute:</p>
                <el-form :model="formDataA">
                    <el-input v-model="formDataA.attr" placeholder="Please enter"  spellcheck="false" type="textarea" style="width:160px;margin-left: 5px;" :autosize="{ minRows: 1, maxRows: 2}">

                    </el-input>

                </el-form>
                <p style="margin-left:10px;user-select: none;">State:</p>
                <el-form :model="formDataA">
                    <el-input v-model="formDataA.state" placeholder="Please enter"  spellcheck="false" type="textarea" style="width:160px;margin-left: 5px;" :autosize="{ minRows: 1, maxRows: 2}">

                    </el-input>

                </el-form>
                <p style="margin-left:10px;user-select: none;">Direction:</p>
                <el-form :model="formDataA">
                    <el-input v-model="formDataA.direction" placeholder="Please enter"  spellcheck="false" type="textarea" style="width:160px;margin-left: 5px;" :autosize="{ minRows: 1, maxRows: 2}">

                    </el-input>

                </el-form>
                <p style="margin-left:10px;user-select: none;">Negative:</p>
                <el-form :model="formDataA">
                    <el-input v-model="formDataA.neg" placeholder="Please enter"  spellcheck="false" type="textarea" style="width:160px;margin-left: 5px;" :autosize="{ minRows: 1, maxRows: 2}">

                    </el-input>

                </el-form>
                <p style="margin-left:10px;user-select: none;">Relationship:</p>
                <!-- <el-select v-model="relationTargetOption" style="width:30px;background-color: aqua;">
                  <el-option
                    v-for="item in TargetOptions"
                    :key="item.value"
                    :label="item.label"
                    :value="item.value"
                  >
                    <span :style="{ backgroundColor: item.color, color:  item.color}">opt</span>
                  </el-option>
                </el-select> -->
                <select v-model="relationTargetOption" :class="selectClass" style="width:25px;height:25px;font-size: 12px;">
                  <option disabled value="">Select a target</option>
                  <option v-for="item in labels" :key="item.value" :value="item.value" :style="{ backgroundColor: item.color, color: item.color }">
                   <span> {{ item.text }} </span>
                  </option>
                </select>
                
                <p></p>
                
                <el-form :model="formDataA">
                    <el-input v-model="formDataA.relation" placeholder="Please enter"  spellcheck="false" type="textarea" style="width:160px;margin-left: 5px;" :autosize="{ minRows: 1, maxRows: 2}">
                    </el-input>
                </el-form>
                <el-button style="margin-top: 10px" @click="postSpaceJson()">
                Confirm
                </el-button>
            </div>
    </div>
</template>
<script>
// import { Search } from "@element-plus/icons-vue";
// import { useRouter } from "vue-router";
// import { dataService } from "@/service";
import { postJson } from "../service/module/dataService";
import { useDataStore } from "../stores/counter";
import mask_prompt from 'E:/CHI2025/repo/RegionDrawing/InterfaceBackend/mask_prompt.json';

// import { ElMessage, ElMessageBox } from "element-plus";
export default {
  name: "APP",
  props: ["msgH"],
  data() {
    return {
      // labels: [
      //   { color: 'red', text: 'boy' },
      //   { color: 'green', text: 'dog' },
      //   { color: 'yellow', text: 'mountain' },
      //   { color: 'blue', text: 'sea' }
      // ],
      labels: [
        { color: 'red', text: 'boy' , value: 'red'},
        { color: 'green', text: 'dog', value: 'green' },
        { color: 'yellow', text: 'mountain' , value: 'yellow'},
        { color: 'blue', text: 'sea' , value: 'blue'}
      ],
      formDataA: {
                type: '',
                attr: '',
                state: '',
                direction: '',
                neg: '',
                relation: '',
            },

      relationTargetOption: '', // This will hold the selected value
      // TargetOptions: [
      //   { value: 'red', label: 'Red', color: 'red' },
      //   { value: 'green', label: 'Green', color: 'green' },
      //   { value: 'blue', label: 'Blue', color: 'blue' },
      //   { value: 'yellow', label: 'Yellow', color: 'yellow' },
      // ],


      // canvas容器
      canvas_bg: undefined,
      canvas_bg_ctx: undefined,
      canvas_bg2: undefined,
      canvas_bg2_ctx: undefined,
      canvas_container: undefined,
      canvas_ctx: undefined,
      textColor: "#000", // 画笔颜色
      lastColor: "#000", // 上一个使用的颜色
      // lineWidth: 5,
      lineWidth: 5,
      do_types: [], // 上次操作的工具 1:笔迹、2:矩阵、3:文字、4:移动矩形、5:移动文本
      imgs: [], // 需要保存的图像
      rect_imgs: [], // 每一步的矩阵
      txt_imgs: [], // 每一步的文字
      img_before_rect: undefined, // 画矩阵前的img
      drawed_rects: [[]], // 画过的矩形 array[{x:x, y:y, w:w, h:h, c:color, lw:linewidth}]
      img_before_txt: undefined, // 编辑txt之前的img

      last_draw_pt: [0, 0], // 上次划过的点
      now_sel_tool_idx: 0, // 目前选中使用的工具
      last_sel_tool_idx: 0, // 上次选中使用的工具
      isDrawing: false, // 当前是否在画画

      default_pen_linewidth: [5, 50, 50], // 默认的普通笔/荧光笔笔触宽度
      // default_pen_linewidth: [50, 50, 50],

      now_sel_rect_idx: -1, // 修改的矩形的编号
      now_sel_rect_direction: "", // 拉伸矩形的方向
      now_sel_rect_former_pos: { x: 0, y: 0 }, // 修改的矩形原本的位置
      txt_input_container: undefined, // 输入文字的文本框
      txt_input_font_family: "Arial", // 输入的文本的字体
      txt_input_size: 20, // 输入的文本字号
      txt_input_content: "", // 输入的文本内容
      txt_input_pos: { x: -1, y: -1 }, // 输入的文本位置
      added_txts: [[]], // 添加的文本内容 {x, y, fontSize, content, color}[][]
      txt_input_timer: 0, // 给结束输入input计时的
      now_sel_txt_idx: -1, // 修改的文本内容的编号
      edit_txt_status: 0, // 0: move, 1: edit content
      now_sel_txt_former_pos: { x: 0, y: 0}, // 修改文本的原本的位置
    };
  },
  computed: {
    selectClass() {
      // return this.relationTargetOption ? `${this.relationTargetOption}-select` : '';
      return this.relationTargetOption ? `${this.relationTargetOption}-select` : '';
    },
    selectedOptionLabel() {
      const selected = this.options.find(option => option.value === this.selectedOption);
      return selected ? selected.label : 'Select a color';
    },
    now_drawing() {
      return (
        (this.now_sel_tool_idx == 1 ||
          this.now_sel_tool_idx == 2 ||
          this.now_sel_tool_idx == 5) &&
        this.isDrawing
      );
    },
    now_recting() {
      return this.now_sel_tool_idx == 3 && this.isDrawing;
    },
    now_selecting_rect() {
      return this.now_sel_tool_idx == 0 && this.now_sel_rect_idx != -1;
    },
    last_edit_rects() {
      return this.drawed_rects[this.drawed_rects.length - 1];
    },
    now_select_rect_pos() {
      if (this.now_sel_tool_idx == 0 && this.now_sel_rect_idx != -1) {
        const now_sel_rect_obj = this.last_edit_rects[this.now_sel_rect_idx];
        return now_sel_rect_obj;
      } else return { x: 0, y: 0, w: 0, h: 0, lineWidth: 5, color: "#000" };
    },
    now_moving_rect() {
      return (
        this.now_sel_tool_idx == 0 &&
        this.now_sel_rect_idx != -1 &&
        this.now_sel_rect_direction != "" &&
        this.isDrawing
      );
    },
    // 当前正在编辑内容
    now_adding_txt() {
      return (
        this.now_sel_tool_idx == 4 &&
        this.txt_input_pos.x >= 0 &&
        this.txt_input_pos.y >= 0
      );
    },
    last_edit_txts() {
      return this.added_txts[this.added_txts.length - 1];
    },
    now_selecting_txt() {
      return this.now_sel_tool_idx == 0 && this.now_sel_txt_idx != -1;
    },
    now_select_txt_pos() {
      if (this.now_sel_tool_idx == 0 && this.now_sel_txt_idx != -1) {
        const now_sel_txt_obj = this.last_edit_txts[this.now_sel_txt_idx];
        return now_sel_txt_obj;
      } else return { x: 0, y: 0, w: 0, fontSize: 20, color: "#000" };
    },
    now_moving_txt() {
      return (
        this.now_sel_tool_idx == 0 &&
        this.now_sel_txt_idx != -1 &&
        this.isDrawing
      );
    },
    // 撤销按钮的样式状态
    backFlag() {
      return this.do_types.length > 0;
    },
  },
  methods: {
    postSpaceJson(){
            let color = this.textColor
            let color2 = this.relationTargetOption
            const dataStore = useDataStore();
            dataStore.type = this.formDataA.type
            dataStore.attribute = this.formDataA.attr
            dataStore.state = this.formDataA.state
            dataStore.direction = this.formDataA.direction
            dataStore.negative = this.formDataA.neg
            dataStore.relationship = this.formDataA.relation
            console.log(this.formDataA.type)
            mask_prompt[color].type = dataStore.type
            mask_prompt[color].attribute = dataStore.attribute
            mask_prompt[color].state = dataStore.state
            mask_prompt[color].direction = dataStore.direction
            mask_prompt[color].negative = dataStore.negative
            if (color2!='')
            {
              mask_prompt['relationships'][color][color2] = dataStore.relationship
            }
            console.log('hey there')
            let _this=this
            postJson(mask_prompt, (res) => {
              // console.log(res);
              console.log(_this.formDataA)
            });
            this.textColor=color
            console.log(this.textColor)
            this.relationTargetOption=color2
            this.changeColor()
        },
    // 改变画笔颜色
    changeColor(event){
      d3.selectAll(".color-rect").style("border", "none")
      // let color = event.target.getAttribute('fill');
      this.relationTargetOption=''
      let color = event.target.style.backgroundColor;
      let color2 = this.relationTargetOption
      this.textColor=color

      event.target.style.border="solid 4px"
      const dataStore = useDataStore();
      dataStore.color = color
      dataStore.type = mask_prompt[color].type
      dataStore.attribute = mask_prompt[color].attribute
      dataStore.state = mask_prompt[color].state
      dataStore.direction = mask_prompt[color].direction
      dataStore.negative = mask_prompt[color].negative
      // if (color2!=''){
      //   dataStore.relation = mask_prompt['relationships'][color][color2]
      // }
      // update_space.methods.updateData();

      // 找到对应颜色的对象并更新 text 属性
      let label = this.labels.find(label => label.color === color);
      if (label) {
        label.text = mask_prompt[color].type;  // 例如，将 text 属性更新为 JSON 中的 type
      } else {
        console.error(`No label found for color: ${colorKey}`);
      }
      // event.target.setAttribute('stroke', "black");
      // event.target.setAttribute('stroke-width', "2px");
    },
    // 是否展示画草图的界面
    show_sketch_pad() {
      this.sketch_pad_show = !this.sketch_pad_show;
    },
    // 清空画布
    clearAll() {
      this.canvas_ctx.clearRect(
        0,
        0,
        this.canvas_container.width,
        this.canvas_container.height
      );
      // 清空画布后重新画一个和画布同样大小的矩形，在保存时可有背景颜色
      this.canvas_ctx.fillStyle = "#fff";
      this.canvas_ctx.fillRect(
        0,
        0,
        this.canvas_container.width,
        this.canvas_container.height
      );
      document.getElementById("back").style.background = "#999";
      this.backFlag = false;
    },
    updateData() {
          console.log('this.formData');
          const dataStore = useDataStore();
          this.formDataA.type = dataStore.type;
          this.formDataA.attr = dataStore.attribute;
          this.formDataA.state = dataStore.state;
          this.formDataA.direction = dataStore.direction;
          this.formDataA.neg = dataStore.negative;
        },
    // 橡皮擦，将画笔变成白色（背景颜色）
    clearSome() {
      // this.pen("#fff");
    },
    // 画笔颜色
    draw(data) {
      this.pen(data);
    },
    // 撤销按钮，将存入的画笔动作数组删除最后一个，再重新绘制到画布上
    back() {
      if (this.backFlag) {
        let img = this.imgs.pop();
        this.canvas_ctx.putImageData(img, 0, 0);
        // 撤销按钮的样式
        if (this.imgs.length > 0) {
          document.getElementById("back").style.background = "#ccc";
          this.backFlag = true;
        } else {
          document.getElementById("back").style.background = "#999";
          this.backFlag = false;
        }
      }
    },
    // 将画布存储为图片显示在页面
    save() {
      document.getElementById("saveImg").src =
        this.canvas_container.toDataURL();
    },

    getCanvasContent() {
      // const canvas = document.getElementById('myCanvas');
      const dataURL = this.canvas_bg.toDataURL();
      console.log(dataURL);
      // 将数据 URL 显示在控制台中
    },
    // 进行涂鸦部分
    draw_content(e) {
      this.canvas_bg_ctx.lineCap = "round";
      this.canvas_bg_ctx.lineWidth = this.lineWidth;
      this.canvas_bg_ctx.beginPath();
      this.canvas_bg_ctx.moveTo(this.last_draw_pt[0], this.last_draw_pt[1]);
      this.canvas_bg_ctx.lineTo(e.offsetX, e.offsetY);
      this.canvas_bg_ctx.stroke();
      this.last_draw_pt = [e.offsetX, e.offsetY];
    },
    start_rect_edit_log() {
      const last_logs = this.drawed_rects[this.drawed_rects.length - 1];
      const new_logs = [];
      last_logs.forEach((rect_info) => {
        new_logs.push({
          x: rect_info.x,
          y: rect_info.y,
          w: rect_info.w,
          h: rect_info.h,
          lineWidth: rect_info.lineWidth,
          color: rect_info.color,
        });
      });
      this.drawed_rects.push(last_logs);
    },
    // 保存当前进度
    save_now_content() {
      // 将每一个画笔动作存入imgs，用于撤销
      if (this.canvas_container.width == 0) return;

      const canvas_list = [
        this.canvas_bg_ctx,
        this.canvas_bg2_ctx,
        this.canvas_ctx,
      ];
      const objs = [];
      const _this = this;
      canvas_list.forEach((canvas_ctx) => {
        objs.push(
          canvas_ctx.getImageData(
            0,
            0,
            _this.canvas_container.width,
            _this.canvas_container.height
          )
        );
      });
      if (this.do_types.length == 0) {
        this.imgs.push(objs[0]);
        this.rect_imgs.push(objs[1]);
        this.txt_imgs.push(objs[2]);
      } else {
        const last_do = this.do_types[this.do_types.length - 1];
        if (last_do == 1) this.imgs.push(objs[0]);
        if (last_do == 2 || last_do == 4) this.rect_imgs.push(objs[1]);
        if (last_do == 3) this.txt_imgs.push(objs[2]);
      }
    },
    save_img_before_recting() {
      const obj = this.canvas_bg2_ctx.getImageData(
        0,
        0,
        this.canvas_container.width,
        this.canvas_container.height
      );
      this.img_before_rect = obj;
    },
    // 撤销按钮，将存入的画笔动作数组删除最后一个，再重新绘制到画布上
    turn_back() {
      if (this.backFlag) {
        const last_do = this.do_types.pop();
        if (last_do == 1) {
          this.imgs.pop();
          // this.steps_img.pop();
          let img = this.imgs[this.imgs.length - 1];
          this.canvas_bg_ctx.putImageData(img, 0, 0);
        } else if (last_do == 2) {
          this.rect_imgs.pop();
          this.last_edit_rects.pop();
          // this.steps_img.pop();
          let img = this.rect_imgs[this.rect_imgs.length - 1];
          this.canvas_bg2_ctx.putImageData(img, 0, 0);
        } else if (last_do == 4) {
          this.rect_imgs.pop();
          this.drawed_rects.pop();
          // this.steps_img.pop();
          let img = this.rect_imgs[this.rect_imgs.length - 1];
          this.canvas_bg2_ctx.putImageData(img, 0, 0);
        } else if (last_do == 3) {
          this.txt_imgs.pop();
          this.last_edit_txts.pop();
          // this.steps_img.pop();
          let img = this.txt_imgs[this.txt_imgs.length - 1];
          this.canvas_ctx.putImageData(img, 0, 0);
        } else if (last_do == 5) {
          this.txt_imgs.pop();
          this.added_txts.pop();
          // this.steps_img.pop();
          let img = this.txt_imgs[this.txt_imgs.length - 1];
          this.canvas_ctx.putImageData(img, 0, 0);
        } 
      }
    },
    sel_sketch_tool(tool_idx) {
      // 选中工具（不同的画笔、文字工具等）
      this.last_sel_tool_idx = this.now_sel_tool_idx;
      this.now_sel_tool_idx = tool_idx;
      if (tool_idx == 1 || tool_idx == 2) {
        this.lineWidth = this.default_pen_linewidth[tool_idx - 1];
        if (this.textColor == "#fff" || this.last_sel_tool_idx == 5) {
          this.textColor = this.lastColor;
        }
      } else if (tool_idx == 5) {
        this.lineWidth = this.default_pen_linewidth[2];
        this.lastColor = this.textColor;
        this.textColor = "#fff";
      } else if (tool_idx == 0) {
        this.now_sel_rect_idx = -1;
        this.now_sel_rect_direction = "";
      }
    },
    // 在画板上画出矩形
    get_now_rect(e) {
      // 获取当钱画的矩阵的位置大小信息
      let width = e.offsetX - this.last_draw_pt[0];
      let height = e.offsetY - this.last_draw_pt[1];
      let startX = this.last_draw_pt[0];
      let startY = this.last_draw_pt[1];
      if (width < 0) {
        startX = e.offsetX;
        width = -width;
      }
      if (height < 0) {
        startY = e.offsetY;
        height = -height;
      }
      return [startX, startY, width, height];
    },
    draw_rect(e) {
      this.canvas_bg2_ctx.putImageData(this.img_before_rect, 0, 0);
      this.canvas_bg2_ctx.strokeRect(...this.get_now_rect(e));
    },
    last_save_rect(e) {
      const now_rect = this.get_now_rect(e);
      const now_rects = this.last_edit_rects;
      now_rects.push({
        x: now_rect[0],
        y: now_rect[1],
        w: now_rect[2],
        h: now_rect[3],
        color: this.textColor,
        lineWidth: this.lineWidth,
      });
    },
    // 判断接触到的矩阵
    handle_canvas_sel_click(e) {
      if (this.now_sel_tool_idx == 0) {
        const txt_sel_res = this.get_sel_txt(e);
        if (!txt_sel_res) {
          this.get_sel_rect(e);
        }
      }
    },
    re_draw_sel_rect() {
      // 重新画被选中的矩形
      // const now_sel_rect_obj = this.now_select_rect_pos;
      const now_rects = this.last_edit_rects;
      const now_sel_rect_obj = now_rects[this.now_sel_rect_idx];
      this.canvas_bg2_ctx.lineWidth = now_sel_rect_obj.lineWidth;
      this.canvas_bg2_ctx.strokeStyle = now_sel_rect_obj.textColor;
      this.canvas_bg2_ctx.strokeRect(
        now_sel_rect_obj.x,
        now_sel_rect_obj.y,
        now_sel_rect_obj.w,
        now_sel_rect_obj.h
      );
    },
    get_sel_rect(e) {
      this.now_sel_rect_idx = -1;
      const now_rects = this.last_edit_rects;
      for (let i = now_rects.length - 1; i >= 0; i -= 1) {
        const now_rect_obj = now_rects[i];
        if (
          e.offsetX >= now_rect_obj.x &&
          e.offsetX <= now_rect_obj.x + now_rect_obj.w &&
          e.offsetY >= now_rect_obj.y &&
          e.offsetY <= now_rect_obj.y + now_rect_obj.h
        ) {
          this.now_sel_rect_idx = i;
        }
      }
      if (this.now_sel_rect_idx != -1) {
        this.canvas_bg2_ctx.clearRect(
          0,
          0,
          this.canvas_container.width,
          this.canvas_container.height
        );
        now_rects.forEach((now_rect, rect_idx) => {
          if (rect_idx != this.now_sel_rect_idx) {
            this.canvas_bg2_ctx.lineWidth = now_rect.lineWidth;
            this.canvas_bg2_ctx.strokeStyle = now_rect.textColor;
            this.canvas_bg2_ctx.strokeRect(
              now_rect.x,
              now_rect.y,
              now_rect.w,
              now_rect.h
            );
          }
        });
        this.save_img_before_recting();
        this.re_draw_sel_rect();
      }
      return this.now_sel_rect_idx != -1;
    },
    start_resize_rect_size(e, target_direction) {
      e.stopPropagation();
      this.isDrawing = true;
      this.now_sel_rect_direction = target_direction;
      this.start_rect_edit_log();
    },
    reset_rect_size(e) {
      // 重新设置矩阵的大小
      // push_direction e s w n es ws en wn
      const push_direction = this.now_sel_rect_direction;
      const canvas_box = this.canvas_container.getBoundingClientRect();
      const mouseX = e.clientX - canvas_box.left;
      const mouseY = e.clientY - canvas_box.top;
      this.canvas_bg2_ctx.putImageData(this.img_before_rect, 0, 0);
      const now_rects = this.last_edit_rects;

      // 获取修改后的位置大小
      const now_obj = now_rects[this.now_sel_rect_idx];
      if (push_direction.includes("e")) {
        if (mouseX >= now_obj.x) now_obj.w = mouseX - now_obj.x;
        else now_obj.w = 0;
      } else if (push_direction.includes("w")) {
        if (mouseX <= now_obj.x + now_obj.w) {
          now_obj.w = now_obj.x + now_obj.w - mouseX;
          now_obj.x = mouseX;
        } else {
          now_obj.x = now_obj.x + now_obj.w;
          now_obj.w = 0;
        }
      }
      if (push_direction.includes("s")) {
        if (mouseY >= now_obj.y) now_obj.h = mouseY - now_obj.y;
        else now_obj.h = 0;
      } else if (push_direction.includes("n")) {
        if (mouseY <= now_obj.y + now_obj.h) {
          now_obj.h = now_obj.y + now_obj.h - mouseY;
          now_obj.y = mouseY;
        } else {
          now_obj.y = now_obj.y + now_obj.h;
          now_obj.h = 0;
        }
      }
      // 重新画矩形
      this.re_draw_sel_rect();
    },
    // 移动矩形的函数
    start_move_rect(e) {
      this.isDrawing = true;
      this.now_sel_rect_direction = "m";
      this.start_rect_edit_log();
      const canvas_box = this.canvas_container.getBoundingClientRect();
      const mouseX = e.clientX - canvas_box.left;
      const mouseY = e.clientY - canvas_box.top;
      this.last_draw_pt = [mouseX, mouseY];

      // 保存矩形原本的位置
      const now_rects = this.last_edit_rects;
      const now_obj = now_rects[this.now_sel_rect_idx];
      this.now_sel_rect_former_pos = { x: now_obj.x, y: now_obj.y };
    },
    move_rect_pos(e) {
      console.log("moving");
      this.canvas_bg2_ctx.putImageData(this.img_before_rect, 0, 0);

      // 获取修改后的位置大小
      const now_rects = this.last_edit_rects;
      const now_obj = now_rects[this.now_sel_rect_idx];
      const canvas_box = this.canvas_container.getBoundingClientRect();
      const mouseX = e.clientX - canvas_box.left;
      const mouseY = e.clientY - canvas_box.top;
      now_obj.x =
        this.now_sel_rect_former_pos.x + (mouseX - this.last_draw_pt[0]);
      now_obj.y =
        this.now_sel_rect_former_pos.y + (mouseY - this.last_draw_pt[1]);
      // 重新画矩形
      this.re_draw_sel_rect();
    },
    // 在开始编辑前保存进度
    start_txt_edit_log() {
      const last_logs = this.added_txts[this.added_txts.length - 1];
      const new_logs = [];
      last_logs.forEach((txt_info) => {
        new_logs.push({
          x: txt_info.x,
          y: txt_info.y,
          w: txt_info.w,
          fontSize: txt_info.lineWidth,
          color: txt_info.color,
        });
      });
      this.added_txts.push(last_logs);
    },
    // 插入文字
    start_insert_txt(e) {
      setTimeout(() => {
        this.txt_input_pos.x = e.offsetX;
        this.txt_input_pos.y = e.offsetY;
        setTimeout(() => {
          this.txt_input_content = "";
          this.txt_input_container.focus();
        }, 100);
      }, 100);
    },
    // 把编辑的文本内容保存
    stay_txt_added() {
      if (this.txt_input_content != "") {
        this.canvas_ctx.font = `${this.txt_input_size}px ${this.txt_input_font_family}`;
        this.canvas_ctx.fillText(
          this.txt_input_content,
          this.txt_input_pos.x,
          this.txt_input_pos.y + this.txt_input_size * 0.9
        );
        this.save_now_content();
        // 保存txt信息
        const now_txts = this.last_edit_txts;
        now_txts.push({
          x: this.txt_input_pos.x,
          y: this.txt_input_pos.y,
          content: this.txt_input_content,
          fontSize: this.txt_input_size,
          w: this.canvas_ctx.measureText(this.txt_input_content).width,
        });
        this.do_types.push(3);
      }
      this.txt_input_content = "";
      this.txt_input_pos = { x: -1, y: -1 };
    },
    // 重新编辑文本内容
    re_add_sel_txt() {
      // 重新画被选中的矩形
      // const now_sel_rect_obj = this.now_select_rect_pos;
      const now_sel_txt = this.last_edit_txts[this.now_sel_txt_idx];
      this.canvas_ctx.font = `${now_sel_txt.fontSize}px ${this.txt_input_font_family}`;
      this.canvas_ctx.fillText(
        now_sel_txt.content,
        now_sel_txt.x,
        now_sel_txt.y + now_sel_txt.fontSize * 0.9
      );
    },
    save_img_before_txting() {
      const obj = this.canvas_ctx.getImageData(
        0,
        0,
        this.canvas_container.width,
        this.canvas_container.height
      );
      this.img_before_txt = obj;
    },
    get_sel_txt(e) {
      this.now_sel_txt_idx = -1;
      const now_txts = this.last_edit_txts;
      for (let i = now_txts.length - 1; i >= 0; i -= 1) {
        const now_txt_obj = now_txts[i];
        if (
          e.offsetX >= now_txt_obj.x &&
          e.offsetX <= now_txt_obj.x + now_txt_obj.w &&
          e.offsetY >= now_txt_obj.y &&
          e.offsetY <= now_txt_obj.y + now_txt_obj.fontSize
        ) {
          this.now_sel_txt_idx = i;
        }
      }
      if (this.now_sel_txt_idx != -1) {
        this.canvas_ctx.clearRect(
          0,
          0,
          this.canvas_container.width,
          this.canvas_container.height
        );
        now_txts.forEach((now_txt, txt_idx) => {
          if (txt_idx != this.now_sel_txt_idx) {
            this.canvas_ctx.font = `${now_txt.fontSize}px ${this.txt_input_font_family}`;
            this.canvas_ctx.fillText(
              now_txt.content,
              now_txt.x,
              now_txt.y + now_txt.fontSize * 0.9
            );
          }
        });
        // 保存需要编辑txt的信息
        const now_sel_txt = this.last_edit_txts[this.now_sel_txt_idx];
        // this.txt_input_content = now_sel_txt.content;
        this.txt_input_pos = { x: now_sel_txt.x, y: now_sel_txt.y };
        this.save_img_before_txting();
        this.re_add_sel_txt();
      }
      return this.now_sel_txt_idx != -1;
    },
    // 移动txt的函数
    start_move_txt(e) {
      this.isDrawing = true;
      this.start_txt_edit_log();
      const canvas_box = this.canvas_container.getBoundingClientRect();
      const mouseX = e.clientX - canvas_box.left;
      const mouseY = e.clientY - canvas_box.top;
      this.last_draw_pt = [mouseX, mouseY];

      // 保存矩形原本的位置
      const now_txts = this.last_edit_txts;
      const now_obj = now_txts[this.now_sel_txt_idx];
      this.now_sel_txt_former_pos = { x: now_obj.x, y: now_obj.y };
    },
    move_txt_pos(e) {
      console.log("moving");
      this.canvas_ctx.putImageData(this.img_before_txt, 0, 0);

      // 获取修改后的位置大小
      const now_rects = this.last_edit_txts;
      const now_obj = now_rects[this.now_sel_txt_idx];
      const canvas_box = this.canvas_container.getBoundingClientRect();
      const mouseX = e.clientX - canvas_box.left;
      const mouseY = e.clientY - canvas_box.top;
      now_obj.x = mouseX;
      now_obj.y = mouseY;
      // 重新画矩形
      this.re_add_sel_txt();
    },
    // 重新设置canvas的大小
    reset_canvass_size(width, height) {
      [
        [this.canvas_container, this.canvas_ctx],
        [this.canvas_bg, this.canvas_bg],
        [this.canvas_bg2, this.canvas_bg2_ctx],
      ].forEach(([container, ctx]) => {
        container.width = width;
        container.height = height;
        ctx.lineWidth = this.lineWidth;
        ctx.strokeStyle = this.textColor;
      });
    },
  },
  created() {},
  mounted() {
    const sketch_pad_box = this.$refs.sketch_pad_all_box;
    this.txt_input_container = this.$refs.text_input_assist;
    // 储存笔迹
    this.canvas_bg = this.$refs.sketch_bg_canvas;
    this.canvas_bg_ctx = this.canvas_bg.getContext("2d");
    // 储存rect
    this.canvas_bg2 = this.$refs.sketch_bg_canvas2;
    this.canvas_bg2_ctx = this.canvas_bg2.getContext("2d");
    // 储存文字
    this.canvas_container = this.$refs.sketch_canvas;
    this.canvas_ctx = this.canvas_container.getContext("2d");

    // 调整canvas的大小
    const _this = this;
    const resizeObserver = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;

      // 重设大小和（设置背景
      // _this.reset_canvass_size(width, height);
      _this.reset_canvass_size(width, height);

      // 设置画笔颜色和宽度
      _this.save_now_content();
    });

    // 开始监听
    resizeObserver.observe(this.canvas_container);

    // 鼠标按下时开始绘画
    _this.canvas_container.addEventListener("mousedown", (e) => {
      // 确认是否是画画工具
      if (
        0 < _this.now_sel_tool_idx &&
        _this.now_sel_tool_idx <= 5 &&
        _this.now_sel_tool_idx != 4
      ) {
        this.isDrawing = true;
        this.last_draw_pt = [e.offsetX, e.offsetY];
      }

      // 如果在画矩阵
      if (_this.now_sel_tool_idx == 3) {
        _this.save_img_before_recting();
      }

      // 如果在输入文字
      _this.start_insert_txt(e);
    });

    // 鼠标移动时绘画
    _this.canvas_container.addEventListener("mousemove", (e) => {
      if (_this.now_drawing) {
        _this.draw_content(e);
      } else if (_this.now_recting) {
        _this.draw_rect(e);
        // 画矩形
      }
    });
    document.addEventListener("mousemove", (e) => {
      // 如果在移动矩形
      if (_this.now_moving_rect) {
        if (_this.now_sel_rect_direction == "m") {
          _this.move_rect_pos(e);
        } else {
          _this.reset_rect_size(e);
        }
      }
      // 如果在移动文本
      if (_this.now_moving_txt) {
        if (_this.edit_txt_status == 0) {
          _this.move_txt_pos(e);
        }
      }
    });

    // 鼠标松开时停止绘画
    function stop_sketch_draw(e) {
      if (_this.now_drawing) {
        _this.do_types.push(1);
        _this.save_now_content();
        _this.isDrawing = false;
      } else if (_this.now_recting) {
        // 正在画矩形
        _this.do_types.push(2);
        _this.save_now_content();
        _this.last_save_rect(e);
        _this.isDrawing = false;
      }
    }
    _this.canvas_container.addEventListener("mouseout", stop_sketch_draw);
    _this.canvas_container.addEventListener("mouseup", stop_sketch_draw);

    function stop_rect_move() {
      if (_this.now_moving_rect) {
        _this.do_types.push(4);
        _this.save_now_content();
        _this.isDrawing = false;
        _this.now_sel_rect_direction = "";
      } if (_this.now_moving_txt) {
        _this.do_types.push(5);
        _this.save_now_content();
        _this.isDrawing = false;
      }
    }
    sketch_pad_box.addEventListener("mouseup", stop_rect_move);
  },
  watch: {
    formData() {
      console.log(this.formData);
    },
    lineWidth(newVal) {
      this.canvas_ctx.lineWidth = newVal;
      this.canvas_bg_ctx.lineWidth = newVal;
      this.canvas_bg2_ctx.lineWidth = newVal;
    },
    textColor(newVal, oldVal) {
      this.lastColor = oldVal;
      this.canvas_ctx.strokeStyle = newVal;
      this.canvas_bg_ctx.strokeStyle = newVal;
      this.canvas_bg2_ctx.strokeStyle = newVal;
      this.updateData()
    },
    do_types: {
      deep: true,
      async handler() {
        // 当画画之后，保存画过的内容
        // let imageBlob;
        await this.canvas_bg.toBlob(blob => {
          // imageBlob = blob;
          dataStore.sketchDrawBlob = blob;
          console.log("blob", blob);
        }, 'image/png');
        // let txtBlob;
        this.canvas_container.toBlob(blob => {
          // txtBlob = blob;
          dataStore.sketchTxtBlob = blob;
        }, 'image/png');
        // let rectBlob;
        this.canvas_bg2.toBlob(blob => {
          // rectBlob = blob;
          dataStore.sketchRectBlob = blob;
        }, 'image/png');
        const dataStore = useDataStore();
        dataStore.sketchRectsInfo = this.last_edit_rects;
        dataStore.sketchTxtsInfo = this.last_edit_txts;
      },
    }
    // do_types() {
    //   console.log("drawed!!!");
    // },
  },
};
</script>
<style scoped>
.sketch_pad_box {
  width: 100%;
  height: 100%;
  position: relative;

  /* overflow-x: hidden; */
  overflow: hidden;
}
.sketch_pad_box canvas {
  width: 100%;
  height: 100%;
}
.sketch_pad_box .sketch_bg_canvas {
  position: absolute;
  top: 0;
  left: 0;
}
.sketch_btns_box {
  position: absolute;
  left: 10px;
  top: 10px;
  /* width: 40px !important; */
  width: 50px;
  /* width: 40px; */
  border-radius: 25px;
  padding: 6px 5px;

  background-color: #f0f0f0;
}
.sketch_btns_box .btn_box {
  padding: 3px;
  /* width: 34px;
  height: 34px; */
  width: 40px;
  height: 40px;
}
.btn_box button {
  width: 100%;
  height: 100%;
  outline: none;
  border: none;
  border-radius: 50%;

  cursor: pointer;
  transition: all ease 300ms;
}

.btn_box button.off_btn path {
  fill: #a0a0a0;
}

.btn_box button:hover {
  background-color: #f6f7f8;
}
.btn_box button.select_btn {
  background-color: #fafbfc;
}

.btn_box svg {
  width: 100%;
  height: 100%;
}

.sketch_btns_box .progress_box {
  padding: 5px 0;
  /* width: 30px; */
  width: 100%;
  height: 100px;
}
.sketch_btns_box .progress_box .el-progress {
  width: 100%;
  height: 100%;
  appearance: slider-vertical;
  margin: 0;
}

.sketch_btns_box .progress_box .el-slider__bar {
  background-color: #3b3b3b; /* 设置进度条颜色 */
}

.sketch_btns_box .progress_box .el-slider__button-wrapper {
  border-color: #001b36; /* 设置滑块边框颜色 */
}
</style>
<style scoped>
.sel_rect_box {
  position: absolute;
  cursor: move;
}
.sel_rect_box > div {
  position: absolute;
  width: 10px;
  height: 10px;
  border: 1px solid #000;
  border-radius: 50%;
  background-color: azure;

  overflow: visible;
}
.sel_rect_box .sel_s,
.sel_rect_box .sel_es,
.sel_rect_box .sel_ws {
  bottom: -5px;
}
.sel_rect_box .sel_n,
.sel_rect_box .sel_en,
.sel_rect_box .sel_wn {
  top: -5px;
}
.sel_rect_box .sel_e,
.sel_rect_box .sel_en,
.sel_rect_box .sel_es {
  right: -5px;
}
.sel_rect_box .sel_w,
.sel_rect_box .sel_ws,
.sel_rect_box .sel_wn {
  left: -5px;
}

.sel_rect_box .sel_s,
.sel_rect_box .sel_n {
  left: calc(50% - 5px);
  cursor: ns-resize;
}
.sel_rect_box .sel_e,
.sel_rect_box .sel_w {
  top: calc(50% - 5px);
  cursor: ew-resize;
}
.sel_rect_box .sel_en,
.sel_rect_box .sel_ws {
  cursor: nesw-resize;
}
.sel_rect_box .sel_es,
.sel_rect_box .sel_wn {
  cursor: nwse-resize;
}
</style>
<style>
#text_input_assist {
  appearance: none;

  position: absolute;
  border: none;
  padding: 0;
  background: transparent;
  /* background-color: rgba(175, 238, 238, 0.5); */
  overflow: hidden;
  outline: none;
  resize: none;
  margin: 0;

  color: #000;
  white-space: nowrap;
}
.txt_edit_btn {
  position: absolute;
  width: 18px;
  height: 18px;
  padding: 2px;
  border: 1.5px solid #000;
  border-radius: 50%;

  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  background-color: #f0f0f0;

  cursor: pointer;
}
.txt_edit_btn svg {
  width: 100%;
  height: 100%;
}

.label-row {
  display: flex;
  align-items: center;
  margin-bottom: 10px;
}

.color-rect {
  width: 20px;
  height: 20px;
  margin-right: 10px;
}


select {
  padding: 10px;
  font-size: 16px;
  border: 1px solid #ccc;
  border-radius: 4px;
  color: white; 
}

.red-select {
  background-color: red;
}

.green-select {
  background-color: green;
}

.blue-select {
  background-color: blue;
}

.yellow-select {
  background-color: yellow;
  color: black; 
}

</style>
