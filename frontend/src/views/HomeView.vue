<template>
  <!-- <div class="common-layout" style="width: 100%; height: 100vh;" v-loading="initSign"
    :element-loading-text="loadingText" element-loading-background="rgba(0, 0, 0, 0.8)"> -->
    <!-- <Main :msgH="msgH"/> -->
    <!-- <el-row :gutter="0" style="height: 100vh">
      <el-col :span="4">
        <div class="grid-content ep-bg-purple" />
      </el-col>
      <el-col :span="16">
        <div class="grid-content ep-bg-purple">
          <Main/>
        </div>
      </el-col>
      <el-col :span="4">
        <div class="grid-content ep-bg-purple" />
      </el-col>
    </el-row> -->
    <Main/>
  <!-- </div> -->
</template>

<script>
import Main from '../components/Main.vue';
import { useDataStore } from "../stores/counter";

export default {
  name: "home_view",
  data() {
    return {
      msgH: null
    };
  },
  computed: {
    initSign() {
      return this.msgH == null;
    },
    loadingText() {
      return "Loading"
      // return "Loading:    " + 'Route: \n' +
      //     !!this.route + '   Station: \n'+ !!this.station + '   Alldata: \n'+ !!this.alldata
    }
  },
  mounted() {
    const dataStore = useDataStore();
    dataStore.fetchHello();
    this.msgH = dataStore.msg;
  },
  methods: {
    fetchData() {
    }
  },
  components: { Main }
};
</script>
<style scoped>
.el-row {
  margin-bottom: 20px;
  /* background-color: rgb(26, 32, 44); */
}
.el-row:last-child {
  margin-bottom: 0;
}
.el-col {
  border-radius: 4px;
}

.grid-content {
  border-radius: 4px;
  min-height: 36px;
}
</style>
