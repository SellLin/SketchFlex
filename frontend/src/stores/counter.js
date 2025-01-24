import { fetchHello, postImg, postText, fetchValue } from "../service/module/dataService";
import { ref, computed } from "vue";
import { defineStore } from "pinia";

// export const useCounterStore = defineStore("counter", {
//   const count = ref(0);
//   const doubleCount = computed(() => count.value * 2);
//   function increment() {
//     count.value++;
//   }

//   return { count, doubleCount, increment };
// });

export const useDataStore = defineStore("dataStore", {
  state: () => {
    return {
      msg: 'Hello, Vue SQ',
      imgFile: '',
      imgSet: [],
      sketchDrawBlob: undefined,
      sketchRectBlob: undefined,
      sketchTxtBlob: undefined,
      sketchRectsInfo: [],
      sketchTxtsInfo: [],
      useSketch: false,
      color: '',
      type:'',
      attribute:'',
      state:'',
      direction:'',
      negative:'',
      relationship:'',
      background:'',
      backgroundAttr:'',
      preview_mask: false,
    }
  },
  actions: {
    fetchHello() {
      const st = new Date();
      fetchHello({}, resp => {
        this.msg = resp;
        console.log("Fetch Hello Time: ", st - new Date());
      })
    },
    postImg({ commit }) {
      const st = new Data();
      postImg({ commit }, resp => {
        // this.imgSet = resp;
        console.log("post Img Time: ", st - new Date());
      })
    },
    postText({ commit }) {
      const st = new Data();
      postText({ commit }, resp => {
        console.log("post Text Time: ", st - new Date());
      })
    },
    fetchValue() {
      const st = new Data();
      fetchValue({}, resp => {
        this.imgSet = resp;
        console.log("Fetch Data Time: ", st - new Date());
      })
    }
  }
})