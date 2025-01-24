<template>
    <!-- <div class="common-layout" style="width: 100%; height: 100vh;" v-loading="initSign"
            :element-loading-text="loadingText" element-loading-background="rgba(0, 0, 0, 0.8)"> -->
    <!-- <Main :msgH="msgH"/> -->
    <Navbar ref="Navbar" @fetchData="fetchData" :message="val"/>
    <!-- {{ val }} -->
    <div class="keywords" style="text-align: start;">
        <el-row>

            <!-- <el-col :span="5" type="flex" justify="left"><span style="font-size: 25px;">{{"Keywords ("+keywordlist.length+"):" }}  </span><span style="margin-left: 30px;">  </span><el-button v-for="(o, index) in keywordlist" color="#626aef" :dark="isDark" plain>{{ o }}</el-button></el-col> -->
            <el-col :span="2" justify="start"><span style="font-size: 25px;margin-left: 52px;">{{ "Elements: " }}
                </span><span style="margin-left: 10px;"> </span>
                <!-- <el-button v-for="(o, index) in keywordlist" size="small" type="keyword"
                    color="gray" :dark="isDark" plain style="font-size: 18px" @click="selectKey($event)">{{
                        o.name.substring(0, 1).toUpperCase() + o.name.substring(1) + " (" + o.count + ")"
                    }}</el-button>
                    <span style="font-size: 25px;margin-left: 52px;">{{ "Elements: " }}
                </span><span style="margin-left: 10px;"> </span> -->


                    <!-- <el-button v-for="(o, index) in elementlist" type="ele"
                    color="gray" :dark="isDark" plain style="font-size: 18px" @click="selectEle($event)">{{
                        o.name.substring(0, 1).toUpperCase() + o.name.substring(1) + " (" + o.count + ")"
                    }}</el-button>  -->

                    
            </el-col>
            <!-- <el-col :span="3" v-for="(o, index) in elementlist" background-color="rgb(204,204,255)">
                <span v-if="index==0||index==2" style="margin-left: 80px;"> </span>
                <el-button  type="ele" size="small"
                    color="gray" :dark="isDark" plain style="font-size: 18px" @click="selectEle($event)">{{
                        o.name.substring(0, 1).toUpperCase() + o.name.substring(1) + " (" + o.count + ")"
                    }}</el-button> 
                <el-select
                    
                    multiple
                    filterable
                    allow-create
                    default-first-option
                    :reserve-keyword="false"
                    placeholder="Related"
                    size="small"
                >
                    <el-option
                    v-for="(s, sindex) in related_taglist.slice(5*index,5*(index+1))"
                    :key="s.name"
                    :label="s.name"
                    :value="s.name"
                    />
                </el-select>
                <el-button v-if="index==0||index==2" type="primary" style="margin-left: 18px;">+</el-button>
            </el-col> -->
            <el-col :span="2.5*composed_element_list[index].length" class="tag_col" v-for="(o, index) in composed_element_list" background-color="rgb(204,204,255)">
                <!-- <span v-if="index==0||index==2" style="margin-left: 80px;"> </span> -->
                <!-- <el-button  type="ele" size="small"
                    color="gray" :dark="isDark" plain style="font-size: 18px" @click="selectEle($event)">{{
                        o.name.substring(0, 1).toUpperCase() + o.name.substring(1) + " (" + o.count + ")"
                    }}</el-button>  -->
                <div v-for="(c, cindex) in composed_element_list[index]" class="tag_div" style="display:inline-block">
                    <!-- <el-slider v-model="val_weight[index][cindex]" style="width:80px" ></el-slider> -->
                <el-select v-model="val[index][cindex]" :placeholder="c.name" class="tag_sel"
                     @change="changeTag" style="width:100px;font-size: 40px;" filterable clearable allow-create
                >
                    <el-option
                    v-for="(s, sindex) in related_taglist[index][0].slice(5*cindex,5*(cindex+1))"
                    :key="s.name"
                    :label="s.name"
                    :value="s.name"
                    />
                </el-select>
                <el-slider v-model="val_weight[index][cindex]" style="width:80px" ></el-slider>
                <!-- <el-button type="primary" size="small" style="margin-left: 18px;">+</el-button> -->
                <span style="margin-left: 20px;"> </span>
            </div>
            </el-col>
            <!-- <el-col :span="2" justify="start"><span style="font-size: 25px;margin-left: 52px;">{{"Price: " }}  </span></el-col> -->
        </el-row>
        <el-row>

            <el-col :span="8" justify="start" algin="top">
                <span style="font-size: 25px;margin-left: 52px; padding-top: -50px;vertical-align: top;">{{ "Price:   " }} </span>
                <span style="margin-top: 20px;margin-left: 18px; width: 220px; height: 50px;" id="pricechart"></span>
            </el-col>
        </el-row>
        <el-row>

            <!-- <el-col :span="13" type="flex" justify="left"><span style="font-size: 25px;margin-left: 0px">Collections:   </span><span style="margin-left: 30px;">  </span><el-button v-for="(o, index) in collections" color="#33A2E4" :dark="isDark" plain>{{ o.id+" ("+o.count+")" }}</el-button></el-col> -->
            <el-col :span="20" justify="start"><span style="font-size: 25px;margin-left: 52px;">Collections: </span><span
                    style="margin-left: 10px;"> </span><el-button v-for="(o, index) in collections" type="collection"
                    color="gray" :dark="isDark" plain style="font-size: 18px" @click="selectCol($event)">{{ o.id +
                        " (" + o.count + ")" }}</el-button></el-col>
            <!-- <el-col :span="4">
                <span style="font-size: 20px;margin-right: 8px;">Price Sort</span>
                <el-radio-group v-model="sortOp">
                <el-radio-button label="no"></el-radio-button>
                <el-radio-button label="high to low"></el-radio-button>
                <el-radio-button label="low to high"></el-radio-button>
                </el-radio-group>
            </el-col> -->
        </el-row>
    </div>
    <br>
    <div id="modal" class="modal-cover">
        <div class="modal">
            <el-button @click="closeModal()" style="float:right"> <el-icon :size="20">
                    <Close />
                </el-icon></el-button>
            <br>
            <div style="width:100%">
                <div style="display:inline-block;float:left">
                    <!-- <img id="selectedImage" />                  -->
                    <canvas id="c1" width="300" height="300"
                    style="border: 1px solid #ccc;margin-left: 30px;margin-top: 20px;"></canvas> <el-button style="margin-top: 30px;margin-left: 30px;" @click="segRetrieve()"> SegRetr</el-button>
                    <el-button style="margin-top: 30px;" @click="segRetrieveNot()"> Reduce </el-button>
                    <el-button style="margin-top: 30px;" @click="segClear()"> Clear </el-button>
                    <!-- <el-button style="margin-top: 30px;" @click="addCompose()">+</el-button> -->
                    <br>
                    <el-button style="margin-top: 5px;margin-left: 30px;" @click="addCompose()">More</el-button>
                    <br>
                    <div class="compose" style="display:none; background-color:#e2e2e2; float:left; width:330px">
                    <el-radio-group v-model="logic" class="compose" style="display:none; margin-left: 10px;">
                        <el-radio-button label="&" size="small">And</el-radio-button>
                        <el-radio-button label="||" size="small">Or</el-radio-button>
                        <el-radio-button label="& !" size="small">Not</el-radio-button>
                        <el-radio-button label="-->" size="small">Change</el-radio-button>
                   </el-radio-group>
                   <br>
                   <el-button ref="img_add_button" class="compose"  @click="submitNewIm($event)" style="margin-left: 30px;">
                        <el-icon :size="25" style="vertical-align: middle">
                            <Camera />
                        </el-icon>
                    </el-button>
                   <input ref="img_add" type="file" name="img" id="new_img" style="display:none" @change="getNewIm($event)">
                   <el-input v-model="add_text" class="compose" :rows="1" type="textarea" placeholder="Text"
                    style="margin-left:0px;width:160px;" />
                    <el-button class="compose" style="display:none" @click="addIntent()">+</el-button>
                    <br>
                    <canvas id="c2" width="300" height="300" class="compose"
                    style="border: 1px solid #ccc;margin-left: 30px;margin-top: 20px;display:none"></canvas> 
                    <el-button @click="closeMore()" class="compose" style="float:right;display:none"> <el-icon :size="10">
                    <Close />
                </el-icon></el-button>
          
                <el-button class="compose" style="margin-top: 30px" @click="segClearNew()"> Clear </el-button>
            </div>
                </div>
                <div
                    style="display:inline-block;text-align: left;float: left;margin-left: 70px;white-space:pre-wrap;width:320px">
                    <el-input v-model="composed_text" :rows="1" type="textarea" placeholder="Modify"
                    style="margin-left:0px;width:440px;margin-top:20px" />
                    <el-button style="margin-top: 10px;" @click="intructChange()"> Preview </el-button>
                    <el-button style="margin-top: 10px;" @click="instructRetrieve()"> Find </el-button>
                    <img id="InstructImage" style="display:none">
                    <p style="display:none"><span class="detailTitle">Collection: </span><span class="detailCon" id="detailCol"></span></p>
                    <p class="deta"><span class="detailTitle">Id: </span><span class="detailCon" id="detailId"></span></p>
                    <p class="deta"><span class="detailTitle">Chain: </span><span class="detailCon" id="detailChain">Ethereum</span></p>
                    <p class="deta"><span class="detailTitle">Address: </span><span class="detailCon" id="detailAd"
                            style="word-wrap:break-word"></span></p>
                    <p class="deta"><span class="detailTitle">Marketplace: </span><span class="detailCon"
                            id="detailMarket">Opensea</span></p>
                    <p style="display:none"><span class="detailTitle">Best offer: </span><span class="detailCon" id="detailOff"></span></p>
                <p class="deta">
                    <!-- <span class="detailTitle">Link: </span> -->
                    <a id="token_link" href="opensea" style="word-wrap:break-word; display:none">Link</a>
                </p>
                <br>
                <div class="deta" id="input_list" style="width:420px;height:50px;border:solid 2px;border-color: rgb(228, 228, 228);border-radius: 10px;">
                    <!-- <p>Hey</p> -->
                </div>
                <el-button class="deta" @click="segRetrieveCompose()">Compose Retr</el-button>
                <!-- <h2>History offer: </h2>
                      <div id="OfferHistory" style="width:300px;height:100px;background-color: rgb(240,240,240);">
                      </div>
                      <br>
                      <h2>Transaction history: </h2>
                      <div id="TracHistory" style="width:300px;height:100px;background-color: rgb(240,240,240);">
                      </div> -->
                </div>
            </div>
        </div>
    </div>
    <div id="modal1" class="modal-cover1">
        <div class="modal1">
            <el-button @click="closeModal1()" style="float:right"> <el-icon :size="20">
                <Close />
            </el-icon></el-button>
        <br>

        <div style="width:100%">
            <!-- <div style="display:inline-block;float:left;width:350px;height:340px;margin-left:10px;">
                     <h2>Prompt</h2>
                     <div></div>
                     <el-input
                        v-model="textarea"
                        :rows="13"
                        type="textarea"
                        placeholder="Please input"
                        style="margin-left:10px"
                    />
                             <el-button @click="generate()">  Generate ✨</el-button>
                        </div>
                        <div style="display:inline-block;float:left;width:360px;height:340px;margin-left:10px;">
                                <img src="https://shadow.elemecdn.com/app/element/hamburger.9cf7b091-55e9-11e9-a976-7f4d0b07eef6.png" width="310" height="305" style="margin-left:30px">
                                <el-button @click="mint()">  Mint </el-button>
                        </div> -->
                <div>
                <div style="position:absolute;margin-left:20px;margin-top:20px;z-index:99999">
                    <img id="genImage" width="460" height="400" style="display:none">
                </div>
                <div style="position:absolute;z-index:-1">
                <canvas id="c" width="440" height="400"
                    style="border: 1px solid #ccc;margin-left: 30px;margin-top: 20px;"></canvas>
                </div>
                
           </div>
                <br>
                <el-input v-model="prompt" :rows="1" type="textarea" placeholder="Please input"
                    style="margin-left:0px;width:440px;margin-top:400px" />
                <el-button @click="submitGenerate()"> Generate ✨</el-button><el-button @click="mint()"> Mint </el-button>
                <el-button @click="clearCanvas($event)">Clear</el-button>
                <!-- <img id="genImage" > -->
        </div>
    </div>
</div>
<div style="" v-loading="!initSign" :element-loading-text="loadingText" element-loading-background="rgba(0, 0, 0, 1)">
        <el-row :gutter="20" style="width: calc(100%);margin-left: 40px;">
            <!-- <el-col :span="4">
                        <div class="grid-content ep-bg-purple">
                            <LeftMenu />
                        </div>
                    </el-col> -->
            <el-col :span="23">
                <div class="grid-content ep-bg-purple">
                    <form id="revise">
                        <input id="annotation" name="text" type="text" style="display: none">
                        <!-- <input type="button" value="Refine" @click="Revise($event)" style="display: block"> -->
                    </form>
                    <!-- <SearchResult :imgSet="imgSet"/> -->
                <div>
                        <el-row :gutter="15">
                            <el-col v-for="(o, index) in imgPath.slice(30 * (Number(radio1) - 1), 30 * (Number(radio1)))"
                                :key="o" :span="2.5" :offset="index % 10 == 0 ? 0 : 0" style="margin-bottom: 20px;">
                                <el-card :body-style="{ padding: '0px' }" style="border-radius: 10px;">
                                    <!-- <img src="https://shadow.elemecdn.com/app/element/hamburger.9cf7b091-55e9-11e9-a976-7f4d0b07eef6.png"
                                                class="image" /> -->
                                    <img :src="getAssetFile(o)" class="image" :id="'image' + index"
                                        @click="showModal($event)"
                                        :price="this.bestOffers[index].substring(0, this.bestOffers[index].length - 4) + ' ETH'"
                                        style="height: 200px;width:221px" />
                                    <div style="padding: 14px;">
                                        <p style="font-size: 13px;text-align: start;"><span
                                                style="font-size: 13px;font-weight: bold;">Id:</span>{{ "" + (o.split("/")[3]=="compressed"?(parseFloat(o.split("/")[7])):parseFloat((o.split("/")[6])))}}
                                        </p>
                                        <!-- <p style="font-size: 15px;text-align: start;"><span
                                                style="font-size: 15px;font-weight: bold;">Id:</span>{{ o.split(" / ")[5] }}
                                        </p> -->
                                        <p style="font-size: 13px;text-align: start;"><span
                                                style="font-size: 13px;font-weight: bold;">Collection:</span><span style="font-size: 10px;word-break: break-all;">{{
                                                    "" + (o.split(" / ")[3] == "compressed" ? o.split("/")[6] : o.split("/")[5]) }}</span></p>
                                    <p style="font-size: 13px;text-align: start;"><span
                                                style="font-size: 13px;font-weight: bold;">Price:</span>{{
                                                    "" + this.bestOffers[index].substring(0, this.bestOffers[index].length - 4) + " ETH" }}
                                        </p>
                                        <div class="bottom">

                                                    <!-- <input type="checkbox" :id="'anno'+index"><span>{{ index }}</span> -->
                                        </div>
                                    </div>
                                </el-card>

                        </el-col>
                    </el-row>
                    </div>
                </div>
            </el-col>
            <!-- <el-col :span="4">
                    <div class="grid-content ep-bg-purple" />
                </el-col> -->
        </el-row>
        <el-row>
            <el-col :span="10"></el-col>
            <el-col :span="4">
                <!-- <el-button text class="button">1</el-button> -->
                <el-radio-group v-model="radio1" class="ml-4">
                    <el-radio-button label="1" size="large">1</el-radio-button>
                    <el-radio-button label="2" size="large">2</el-radio-button>
                    <el-radio-button label="3" size="large">3</el-radio-button>
                    <el-radio-button label="4" size="large">4</el-radio-button>
                    <!-- <el-radio-button v-for"(o, index)"></el-radio-button> -->
                </el-radio-group>
            </el-col>
            <el-col :span="10"></el-col>
        </el-row>
        <!-- </div> -->
    </div>
</template>


<script>

import Main from '../components/Main.vue';
// import Navbar from '../components/Navbar.vue';
import LeftMenu from '../components/LeftMenu.vue';
import { useDataStore } from "../stores/counter";
import SearchResult from '../components/SearchResult.vue';
// import { fetchValue, reviseText, postGenerate, postSam, postText, postImgSeg, postImgSegNeg,postSamClear, postInstruct, postInsRetr, postSamNew, postSamClearNew, postCompose } from '../service/module/dataService'
import { ref } from 'vue'
// const val = ref('')
import {
    Check,
    Delete,
    Edit,
    Message,
    Search,
    Star,
} from '@element-plus/icons-vue'
// const radio1 = ref('1')
export default {
    name: "home_view",
    data() {
        return {
            imgSet: null,
            allImgSet: [],
            imgPath: [],
            imgTag: [],
            modal_src: '',
            allprices: [],
            bestOffers: [],
            priceNumbers: [],
            keywordlist: [],
            elementlist: [],
            composed_element_list: [],
            related_taglist: [],
            textQuery: "",
            Allcount: null,
            selected: null,
            page: 0,
            gval: ["what?????","shshjh"],
            val: [],
            val_weight:[],
            lower:null,
            upper:null,
            bins:null,
            x:null,
            prompt:"",
            currentcol:[],
            currentkey:[],
            radio1: 1,
            logic: "&",
            newIm: '',
            newImgURL: '',
            add_text: '',
            col_Tag: [],
            keycount: [],
            extra_intents: [],
            extra_intent_logic: [],
            extra_intent_modal: [],
            canvas: null,
            canvas1: null,
            canvas2: null,
            upPointNew: null,
            downPointNew: null,
            downPrecordNew: null,
            upPrecordNew: null,
            downPoint: null,
            upPoint: null,
            downPrecord: null,
            upPrecord: null,
            samImPath: null,
            segImPath: null,
            segImPathNew: null,
            segImPathNewTemp: null,
            newImChange: 0,
            ins_segImPath: null,
            segOriImPath: null,
            composed_text: "",
            sortOp: "",
            contract_address: [],
            collections: [
                {
                    id: 'CryptoPunks',
                    type: true,
                    count: 10
                },
                {
                    id: 'Bored Ape Yacht Club',
                    type: true,
                    count: 20
                },
                {
                    id: 'Azuki',
                    type: true,
                    count: 30
                },
                {
                    id: 'Mutant Ape Yacht Club',
                    type: true,
                    count: 10
                },
                {
                    id: '0N1 Force',
                    type: true,
                    count: 20
                },
                {
                    id: 'Doodles',
                    type: true,
                    count: 10
                },
                {
                    id: '3Landers',
                    type: true,
                    count: 15
                },

            ]
        };
    },
    computed: {
        initSign() {
            return this.imgSet != null;
        },
        loadingText() {
            return "Loading"
            // return "Loading:    " + 'Route: \n' +
            //     !!this.route + '   Station: \n'+ !!this.station + '   Alldata: \n'+ !!this.alldata
        }
    },
    mounted() {
        // const dataStore = useDataStore();
        // dataStore.fetchValue();
        // this.imgSet = dataStore.imgSet;
        // fetchValue({}, res => {
        //     console.log(res)
        //     this.imgSet = res;
        //     let imgP = [];
        //     for (let i in this.imgSet) {

        //         let tp;
        //         if (this.imgSet[i].charAt(0)!="s")
        //         {tp= this.imgSet[i].slice(1);}
        //         else{
        //             tp= "/"+this.imgSet[i];
        //         }
        //         let tend = '../assets' + tp;
        //         imgP.push(tend);
        //     }
        //     this.imgPath = imgP;
        //     // this.imgSet=null;
        // })
        this.fetchData();
        this.canvas = new fabric.Canvas('c',{backgroundVpt: false})
        this.canvas1= new fabric.Canvas('c1')
        this.canvas2= new fabric.Canvas('c2')
        // 将画布设置成绘画模式
        this.canvas.isDrawingMode = true
        // console.log("!!!!!!!!!!!!!!!!");
        // console.log(this.$route.name);
    },
    methods: {
        fetchData() {
            // console.log("fetchData")
            fetchValue({}, res => {
                // console.log(res)
                this.allImgSet = res
                // console.log(res)
                let Allcount = 0
                let cols = [];
                let colcount = [];
                let collection_list = [];
                for (let p in this.allImgSet) {
                    Allcount = Allcount + 1
                }
                let tempset = []
                let size = Math.min(100, Allcount)
                for (let i = 0; i < size; i++) {
                    tempset.push(this.allImgSet[i])
                }
                this.imgSet = tempset
                // this.imgSet = this.allImgSet.slice(0,100);
                let imgP = [];
                let bestOff = [];
                let priceNums = [];
                let keycounts = [];
                let col_ads = []
                // console.log(this.imgSet[0])
                // console.log(this.imgSet[1][0])
                if (this.imgSet[1][0].split("#").length > 0) {
                    let first = this.imgSet[0]
                    // console.log(this.imgSet)
                    let keywords0 = first.split("#")[2]
                    console.log("keywords")
                    // console.log(keywords0)
                    let keywords = keywords0.split("----")[0]
                    let elements=keywords0.split("----")[1]
                    let relatedTags=keywords0.split("----")[2]
                    let textQuery=keywords0.split("----")[3]
                    let keytemp = keywords.split(',')
                    // let eletemp = elements.split(',')
                    let elegroup=elements.split('****')
                    let ele_matrix=[]
                    let val_matrix=[]
                    let slide_matrix=[]
                    for (let e=0;e<elegroup.length;e++)
                    {
                        if (elegroup[e]!="")
                        {
                            let eletemp=elegroup[e].split(',')
                            let elelist = []
                            let vallist = []
                            let slidelist = []
                            for (let j = 0; j < eletemp.length; j++) {
                                if (eletemp[j] != "" && eletemp[j] != " ") {
                                    elelist.push({ name: eletemp[j], count: 0 })
                                    vallist.push('')
                                    slidelist.push(Number(50))
                                    // keycounts.push(0)
                                }
                            }
                            if (elelist.length>0)
                            {
                                ele_matrix.push(elelist)
                                val_matrix.push(vallist)
                                slide_matrix.push(slidelist)
                            }
                        }
                    }
                    console.log(relatedTags)
                    let taggroup=relatedTags.split('++++')
                    let tag_matrix=[]
                    for (let e=0;e<taggroup.length;e++)
                    {
                        let tagGroupList=[]
                        if (taggroup[e]!="")
                        {
                            let tagtemp=taggroup[e].split(',')
                            let taglist = []
                            for (let j = 0; j < tagtemp.length; j++) {
                                if (tagtemp[j] != "" && tagtemp[j] != " ") {
                                    taglist.push({ name: tagtemp[j], count: 0 })
                                    // keycounts.push(0)
                                }
                            }
                            // tag_matrix.push(taglist)
                            if (taglist.length>0)
                            {
                                tagGroupList.push(taglist)
                            }
                        }
                        if (tagGroupList.length>0)
                        {
                            tag_matrix.push(tagGroupList)
                        }
                    }
                    // console.log(tag_matrix)
                    // let tagtemp = relatedTags.split(',')
                    let keylist = []
                    for (let j = 0; j < keytemp.length; j++) {
                        if (keytemp[j] != "" && keytemp[j] != " ") {
                            keylist.push({ name: keytemp[j], count: 0 })
                            keycounts.push(0)
                        }
                    }
                   
                    // for (let j = 0; j < eletemp.length; j++) {
                    //     if (eletemp[j] != "" && eletemp[j] != " ") {
                    //         elelist.push({ name: eletemp[j], count: 0 })
                    //         // keycounts.push(0)
                    //     }
                    // }
                    // let taglist = []
                    // for (let j = 0; j < tagtemp.length; j++) {
                    //     if (tagtemp[j] != "" && tagtemp[j] != " ") {
                    //         taglist.push({ name: tagtemp[j], count: 0 })
                    //         // keycounts.push(0)
                    //     }
                    // }
                    
                    this.keywordlist = keylist
                    // this.elementlist = elelist
                    this.composed_element_list=ele_matrix
                    this.related_taglist = tag_matrix
                    this.textQuery=textQuery
                    this.val=val_matrix
                    this.val_weight=slide_matrix
                    this.$refs.Navbar.formData.name=textQuery
                    console.log(textQuery)
                    console.log("check_ele_tag")
                    console.log(ele_matrix)
                    console.log(tag_matrix)
                }
                let tags = []
                // console.log("alllenth")
                // console.log(this.allImgSet.length)

                this.Allcount = Allcount
                let coltags = []
                let allprs = []
                for (let i = 0; i < Allcount; i++) {
                    tags.push(this.keywordlist[i % (this.keywordlist.length)].name)
                    this.keywordlist[i % (this.keywordlist.length)].count = this.keywordlist[i % (this.keywordlist.length)].count + 1
                    // let el=this.allImgSet[i]
                    let tp;
                    let s = this.allImgSet[i]
                    // console.log(this.imgSet[i])
                    let imp = s.split("#")[0]
                    let price = s.split("#")[1]
                    allprs.push(Number(price.split(' ')[0]))
                    if (imp.charAt(0) != "s") { tp = imp.slice(1); }
                    else {
                        tp = "/" + imp;
                    }
                    let el = '../assets' + tp;
                    // console.log(el)
                    let co = (el.split("/")[3] == "compressed" ? el.split("/")[4] : el.split("/")[3])
                    coltags.push(co)
                    let colidx = cols.indexOf(co)
                    if (colidx == -1) {
                        cols.push(co)
                        colcount.push(1)
                        // if (co.substring(0,1)==" "){
                        //     console.log(co)
                        // }
                        // console.log(co.substring(0,2))
                    }
                    else {
                        colcount[colidx] = colcount[colidx] + 1
                    }
                }
                // console.log("tags:")
                // console.log(tags)
                this.keycount = keycounts
                this.imgTag = tags
                this.col_Tag = coltags
                this.allprices = allprs
                // this.contract_address=col_ads
                // for (let i=0;i<Allcount;i++)
                // {

                // }
                for (let i = 0; i < cols.length; i++) {
                    collection_list.push({ id: cols[i], count: colcount[i] })
                }
                this.collections = collection_list
                // console.log("image_set")
                // console.log(this.imgSet[0])
                let len = this.imgSet.length
                for (let i = 0; i < len; i++) {

                    let tp;
                    let s = this.imgSet[i]
                    // console.log(s.split("#").length)
                    // console.log(this.imgSet[i])
                    // console.log(s)
                    if (s.includes("#")) {
                        let imp = s.split("#")[0]
                        let price = s.split("#")[1]
                        // console.log(s.split("#")[3])
                        let col_ad = s.split("#")[3]
                        col_ads.push(col_ad)
                        if (imp.charAt(0) != "s") { tp = imp.slice(1); }
                        else {
                            tp = "/" + imp;
                        }
                        let ex_p="https://nft-1259172767.cos.ap-guangzhou.myqcloud.com/"
                        // console.log(tp.substring(7))
                        let tend = '../assets' + tp;
                        // console.log(tend)
                        // imgP.push(tend);
                        imgP.push(ex_p+tp.substring(7))
                        bestOff.push(price)
                        priceNums.push(Number(price.split(' ')[0]))
                    }
                }
                this.imgPath = imgP;
                // console.log(this.imgPath)
                this.bestOffers = bestOff;
                this.priceNumbers = priceNums;
                this.contract_address = col_ads
                // for (let i=0;i<40;i++)
                // {

                //     document.getElementById("anno"+i.toString()).checked=false
                // }
                // this.imgSet=null;
                // let upper=Math.max.apply(null,this.priceNumbers);
                // let lower=Math.min.apply(null,this.priceNumbers);
                let upper = Math.max.apply(null, this.allprices);
                let lower = Math.min.apply(null, this.allprices);
                this.lower=lower
                this.upper=upper
                d3.select("#priceHis").remove();
                let margin = { top: 2, right: 3, bottom: 1, left: 1 },
                    width = 300 - margin.left - margin.right,
                    height = 40 - margin.top - margin.bottom;
                let svg = d3.select("#pricechart").append("svg")
                    .attr("width", width + margin.left + margin.right)
                    .attr("height", height + margin.top + margin.bottom)
                    .attr("id", "priceHis")
                    .append("g")
                    .attr("transform",
                        "translate(" + margin.left + "," + margin.top + ")");
                // X axis: scale and draw:
                let x = d3.scaleLinear()
                    .domain([lower, upper])     // can use this instead of 1000 to have the max of data: d3.max(data, function(d) { return +d.price })
                    .range([0, width]);
                this.x=x
                svg.append("g")
                    .attr("transform", "translate(0," + height + ")")
                    .attr("class", "axisRed")
                    .call(d3.axisBottom(x));




                // set the parameters for the histogram
                let histogram = d3.histogram()
                    .value(function (d) { return d; })   // I need to give the vector of value
                    .domain(x.domain())  // then the domain of the graphic
                    .thresholds(x.ticks(10)); // then the numbers of bins
                // And apply this function to data to get the bins
                // let bins = histogram(this.priceNumbers);
                let bins = histogram(this.allprices);

                // Y axis: scale and draw:
                let y = d3.scaleLinear()
                    .range([height, 0]);
                y.domain([0, d3.max(bins, function (d) { return d.length; })]);   // d3.hist has to be called before the Y axis obviously
                // svg.append("g")
                //     .attr("class","axisRed")
                //     .call(d3.axisLeft(y).ticks(2));
                // console.log(bins)
                // append the bar rectangles to the svg element
                svg.selectAll("rect")
                    .data(bins)
                    .enter()
                    .append("rect")
                    .attr('id', (d, i) => 'pr' + i)
                    .attr("x", 1)
                    .attr("transform", function (d) { return "translate(" + x(d.x0) + "," + y(d.length) + ")"; })
                    .attr("width", function (d) { return x(d.x1) - x(d.x0) - 8; })
                    .attr("height", function (d) { return height - y(d.length); })
                    .style("fill", "#D2D2D2")
                let vm=this
                const priceBrush = d3.brushX()
                    .extent([[0, 0], [width, height]])
                    // .on("brush end", brushended)
                //   .on("end", brushended);

                const gb = svg.append('g')
                    .attr('id', 'price_brush')
                    .call(priceBrush)
                    .call(priceBrush.move, [0, width + margin.left + margin.right]);

                // var all_prices=this.allprices
                // var all_count=this.Allcount
                
                function brushended({ selection }) {
                    let newImgSet = []
                    let imgP = []
                    let bestOff = []
                    let priceNums = []
                    let indexes=[]
                    for (let i in bins) {
                        if ((x(bins[i].x0) >= selection[0] && x(bins[i].x1) <= selection[1])) {
                            // console.log("#pr" + i);
                            d3.select("#pr" + i).style('fill', 'red')

                            // console.log("this is this")
                            // console.log("this is this")
                            // console.log(vm.Allcount)
                            for (let u = 0; u < vm.Allcount; u++) {
                                //  console.log(vm.allprices[u])
                                //  console.log(bins[i].x0)
                                if (vm.allprices[u]>bins[i].x0 && vm.allprices[u]<=bins[i].x1 ) {
                                    newImgSet.push(vm.allImgSet[u])
                                    // indexes.push(u)
                                }
                            }
                            vm.imgSet = newImgSet
                            // console.log(vm.imgSet.length)
                            for (let u = 0; u < vm.imgSet.length; u++) {
                                let tp;
                                let s = vm.imgSet[u]//[0]
                                let imp = s.split("#")[0]
                                let price = s.split("#")[1]
                                if (imp.charAt(0) != "s") { tp = imp.slice(1); }
                                else {
                                    tp = "/" + imp;
                                }
                                let tend = '../assets' + tp;
                                imgP.push(tend);
                                bestOff.push(price)
                                priceNums.push(Number(price.split(' ')[0]))
                            }

                        }
                        else {
                            d3.select("#pr" + i).style('fill', '#D2D2D2')

                        }
                    }
                    // if (imgP.length>0)
                    // {
                    //     vm.imgPath = imgP;
                    //     console.log(imgP)
                    //     vm.bestOffers = bestOff;
                    //     vm.priceNumbers = priceNums;
                    // }
                    vm.imgPath = imgP;
                    // console.log(imgP)
                    vm.bestOffers = bestOff;
                    vm.priceNumbers = priceNums;
                }


                //画价格变化数据line chart, 后面补全数据后应当从一个csv导入数据而非生成数据
                let priceTrends = []
                let timeStep = 12
                for (let i = 0; i < timeStep; i++) {
                    priceTrends.push(0)
                }
                for (let i = 0; i < timeStep; i++) {
                    for (let j = 0; j < priceNums.length; j++) {
                        priceTrends[i] = priceTrends[i] + priceNums[j] + 0.5 * Math.random() - 0.5
                    }
                }
                for (let i = 0; i < timeStep; i++) {
                    priceTrends[i] = priceTrends[i] / priceNums.length
                }
                // d3.select("#priceLine").remove();
                // let svg1=d3.select("#chart").append("svg")
                // .attr("width", width + margin.left + margin.right)
                // .attr("height", height + margin.top + margin.bottom)
                // .attr("id","priceLine")
                // .append("g")
                // .attr("transform",
                //     "translate(" + margin.left + "," + margin.top + ")");
            })
        },
        getAssetFile(url) {
            return new URL(url, import.meta.url).href;
        },
        changeTag(){
            console.log(this.val)
        },
        nftSearch() {
            const dataStore = useDataStore();
            // Image
           if (this.file == '' && this.formData.name != '') {
                let formData = new FormData();
                formData.append('text', this.formData.name);
                // this.$route.params.search = this.formData.name;
                if (!this.$route.params.search) {
                    this.$route.params.search = ' '
                }
                // let tags = ""
                // let tl_col = d3.selectAll(".tag_col")
                // let tl_sel = tl_col.selectAll(".tag_sel")
                // // let tl_el = tl_sel._groups[0][0]
                // console.log("fkk")
                console.log(this.message[0])
                formData.append('tag', JSON.stringify(this.message))
                postText(formData, res => {
                    console.log("postTextRes")
                    console.log(res);
                    this.fetchData();
                });
                // if()
                // this.jump();
                console.log("###################");
                console.log(this.$route)
                if (this.formData.name != this.$route.params.search) {
                    this.$router.replace('result' + '?test=' + this.formData.name);
                    this.$route.params.search = this.formData.name;
                }
                // else {

                // }
                console.log(this.$route.params)

            }
            // this.$emit("fetchData")
        },
        clearCanvas(event){
            this.canvas.clear();
        },
        canvasToBase64(canvas){
            // 'image/png'可以换成'image/jpeg'
            return canvas.toDataURL('image/png');
        },
        base64ToFile(urlData, fileName){
            let arr = urlData.split(',');
            let mime = arr[0].match(/:(.*?);/)[1];
            let bytes = atob(arr[1]);
            let n = bytes.length
            let ia = new Uint8Array(n);
            while (n--) {
                ia[n] = bytes.charCodeAt(n);
            }
            return new File([ia], fileName, { type: mime });
        },
        intructChange(){
            let formData= new FormData();
            formData.append("im_p", this.samImPath);
            formData.append('composed', this.composed_text);
            postInstruct(formData, res => {
                document.getElementById("InstructImage").src="./src/assets/static/temp_ins/"+res["address"]
                document.getElementById("InstructImage").style="display:block"
                this.ins_segImPath=res["address"]
            
            });
            d3.selectAll(".deta").style("display", "none")
        },
        instructRetrieve(){
            let formData= new FormData();
            formData.append("ins_seg_path", this.ins_segImPath);
            postInsRetr(formData, res => {
                console.log(res);
                this.fetchData();
            });
            let modal = document.getElementById('modal')
            modal.style.display = 'none'
            
            d3.select("#InstructImage").style("display","none")
            d3.selectAll(".deta").style("display","block")
        },
        submitNewIm(event) {
            this.$refs.img_add.dispatchEvent(new MouseEvent('click'))

        },
        getNewIm(event){

            console.log(event.target.files);
            this.newIm = event.target.files[0];
            console.log(this.file);
            this.newImChange=1

            event.preventDefault();
            let URL = window.URL || window.webkitURL;
            let imgURL = URL.createObjectURL(this.newIm);
            this.newImgURL = imgURL;
            let canvas=this.canvas2
            let im_src2=this.newImgURL
            canvas.clear();
            // this.segOriImPath=im_src1
            fabric.Image.fromURL(
                im_src2, // 图片路径
                img => {
                // img.top = 0
                // img.left = 0
                // img.bottom =300
                // img.right =300
                // canvas.add(img) // 将图片插入到画布中
                // canvas.setBackgroundImage(img)
                img.set({
                    // 通过scale来设置图片大小，这里设置和画布一样大
                    scaleX: canvas.width / img.width,
                    scaleY: canvas.height / img.height,
                })
                // 设置背景
                canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas))

                canvas.renderAll()
                }
            )
            // canvas.isDrawingMode = true
            canvas.on('mouse:down', this.canvasMouseDownNew)   // 鼠标在画布上按下
            canvas.on('mouse:up', this.canvasMouseUpNew)       // 鼠标在画布上松开
        },
        submitGenerate(){
            let canvas=this.canvas
            let formData= new FormData();
            formData.append("image", this.base64ToFile(canvas.toDataURL('image/png')),"image");
            let prtext=this.prompt
            // if (this.isCanvasBlank(this.canvas))
            // {
            //     prtext=prtext+"##0"
            // }
            // else
            // {
            //     prtext=prtext+"##1"
            // }
            formData.append("text", prtext);
            console.log("problem?")
            // var img = new Image();
            // var genpath;
            postGenerate(formData, res => {
                // console.log(res)
                // console.log("../assets/"+res["address"])
                document.getElementById("genImage").src="./src/assets/"+res["address"]
                // document.getElementById("c").style.display="none"
                // document.getElementById("c").style.height="0"
                document.getElementById("genImage").style="display:block"
                // let ctx=c.getContext("2d");
                // let img=document.getElementById("genImage");
                // ctx.drawImage(img,30,30);
                
			//     img.onload = function(){
			// 	alert('加载完毕')
				
			// 	// 将图片画到canvas上面上去！
			// 	ctx.drawImage(img,100,100);
 
            //     genpath="../assets/"+res["address"]
			// }
            
            // console.log(genpath)
			// img.src ="./src/assets/"+res["address"]
            // ctx.drawImage(img,100,100);
            
            });
        },
        segRetrieve(){
            let formData = new FormData();
            formData.append('seg_path', this.segImPath);
            formData.append('composed', this.composed_text);
            postImgSeg(formData, res => {
                console.log(res);
                this.fetchData();
            });
            let modal = document.getElementById('modal')
            modal.style.display = 'none'
        },
        segRetrieveNot(){
            let formData = new FormData();
            formData.append('seg_path', this.segImPath);
            formData.append('composed', this.composed_text);

            postImgSegNeg(formData, res => {
                console.log(res);
                this.fetchData();
            });
            let modal = document.getElementById('modal')
            modal.style.display = 'none'
        },
        segRetrieveCompose(){
            let formData = new FormData();
            formData.append('seg_path', this.segImPath);
            // formData.append('composed', this.composed_text);
            formData.append('extra_intent', this.extra_intents[0])
            formData.append('extra_logic', this.extra_intent_logic[0])
            formData.append('extra_type', this.extra_intent_modal[0])

            postCompose(formData, res => {
                console.log(res);
                this.fetchData();
            });
            let modal = document.getElementById('modal')
            modal.style.display = 'none'
        },
        segClear(){
            let formData= new FormData();
            postSamClear(formData, res => {
                let canvas=this.canvas1
                canvas.clear()
                let im_src1=this.segOriImPath
                fabric.Image.fromURL(
                im_src1, // 图片路径
                img => {
                // img.top = 0
                // img.left = 0
                // img.bottom =300
                // img.right =300
                // canvas.add(img) // 将图片插入到画布中
                // canvas.setBackgroundImage(img)
                img.set({
                    // 通过scale来设置图片大小，这里设置和画布一样大
                    scaleX: canvas.width / img.width,
                    scaleY: canvas.height / img.height,
                })
                // 设置背景
                canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas))

                canvas.renderAll()
                      }
                )
            })
        },
        segClearNew(){
            let formData= new FormData();
            postSamClearNew(formData, res => {
                let canvas=this.canvas2
                canvas.clear()
                let im_src1=this.newImgURL
                fabric.Image.fromURL(
                im_src1, // 图片路径
                img => {
                // img.top = 0
                // img.left = 0
                // img.bottom =300
                // img.right =300
                // canvas.add(img) // 将图片插入到画布中
                // canvas.setBackgroundImage(img)
                img.set({
                    // 通过scale来设置图片大小，这里设置和画布一样大
                    scaleX: canvas.width / img.width,
                    scaleY: canvas.height / img.height,
                })
                // 设置背景
                canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas))

                canvas.renderAll()
                      }
                )
            })
        },
        submitRegion(){
            let formData= new FormData();
            formData.append("up_x", this.upPrecord.x)
            formData.append("up_y", this.upPrecord.y)
            formData.append("down_x", this.downPrecord.x)
            formData.append("down_y", this.downPrecord.y)
            formData.append("im_p", this.samImPath)
            postSam(formData, res => {
                console.log(res)
                let canvas=this.canvas1
                canvas.clear();
                let im_src1="./src/assets/mask/"+(res["address"].split(".")[0])+"comb_v.png"
                this.segImPath=res["address"]
                fabric.Image.fromURL(
                    im_src1, // 图片路径
                    img => {
                    // img.top = 0
                    // img.left = 0
                    // img.bottom =300
                    // img.right =300
                    // canvas.add(img) // 将图片插入到画布中
                    // canvas.setBackgroundImage(img)
                    img.set({
                        // 通过scale来设置图片大小，这里设置和画布一样大
                        scaleX: canvas.width / img.width,
                        scaleY: canvas.height / img.height,
                    })
                    // 设置背景
                    canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas))

                    canvas.renderAll()
                    }
                )
            
            });
        },
        submitRegionNew(){
            let formData= new FormData();
            formData.append("up_x", this.upPrecordNew.x)
            formData.append("up_y", this.upPrecordNew.y)
            formData.append("down_x", this.downPrecordNew.x)
            formData.append("down_y", this.downPrecordNew.y)
            // formData.append("im_p", this.samImPath)
            formData.append("image", this.newIm)
            postSamNew(formData, res => {
                console.log(res)
                let canvas=this.canvas2
                canvas.clear();
                let im_src2="./src/assets/mask/"+(res["address"].split(".")[0])+"comb_v.png"
                this.segImPathNew=res["address"]
                this.segImPathNewTemp=im_src2
                fabric.Image.fromURL(
                    im_src2, // 图片路径
                    img => {
                    // img.top = 0
                    // img.left = 0
                    // img.bottom =300
                    // img.right =300
                    // canvas.add(img) // 将图片插入到画布中
                    // canvas.setBackgroundImage(img)
                    img.set({
                        // 通过scale来设置图片大小，这里设置和画布一样大
                        scaleX: canvas.width / img.width,
                        scaleY: canvas.height / img.height,
                    })
                    // 设置背景
                    canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas))

                    canvas.renderAll()
                    }
                )
            
            });
        },
        isCanvasBlank(canvas) {
            var blank = document.createElement('canvas');//系统获取一个空canvas对象
            blank.width = canvas.width;
            blank.height = canvas.height;
            return canvas.toDataURL() == blank.toDataURL();//比较值相等则为空
        },
        showModal(event) {
            let modal = document.getElementById('modal')
            
            // document.getElementById('selectedImage').src = event.target.src
            let im_src=event.target.src
            this.modal_src=im_src
            this.samImPath=im_src
            let im_src1=this.modal_src
            // console.log(this.contract_address)
            // console.log(event.target.id.substring(5))
            let ad = this.contract_address[Number(event.target.id.substring(5))]
            // console.log(event.target.getAttribute("price"))
            let path = (event.target.src)
            let ts = path.split("/")
            let tn = ts[ts.length - 1]
            let t_id = tn.split(".")[0]
            let pa = event.target.src
            let col = pa.split("/")[7]
            // console.log(t_id)
            document.getElementById('token_link').href = "https://opensea.io/assets/ethereum/" + ad + "/" + t_id
            document.getElementById('token_link').innerText = "https://opensea.io/assets/ethereum/" + ad + "/" + t_id
            document.getElementById('detailCol').innerText = col
            document.getElementById('detailId').innerText = t_id
            document.getElementById('detailAd').innerText = ad
            document.getElementById('detailOff').innerText = event.target.getAttribute("price")
            modal.style.display = 'block'
            // add canvas for SAM
            // let canvas = new fabric.Canvas('c1', {
            //     width: 340,
            //     height: 300
            // })
            let canvas=this.canvas1
            canvas.clear();
            this.segOriImPath=im_src1
            fabric.Image.fromURL(
                im_src1, // 图片路径
                img => {
                // img.top = 0
                // img.left = 0
                // img.bottom =300
                // img.right =300
                // canvas.add(img) // 将图片插入到画布中
                // canvas.setBackgroundImage(img)
                img.set({
                    // 通过scale来设置图片大小，这里设置和画布一样大
                    scaleX: canvas.width / img.width,
                    scaleY: canvas.height / img.height,
                })
                // 设置背景
                canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas))

                canvas.renderAll()
                }
            )
            // canvas.isDrawingMode = true
            canvas.on('mouse:down', this.canvasMouseDown)   // 鼠标在画布上按下
            canvas.on('mouse:up', this.canvasMouseUp)       // 鼠标在画布上松开
        },
        createRect() {
            // 如果点击和松开鼠标，都是在同一个坐标点，不会生成矩形
            if (JSON.stringify(this.downPoint) === JSON.stringify(this.upPoint)) {
                return
            }
            let canvas=this.canvas1
            let downPoint=this.downPoint
            let upPoint=this.upPoint
            // 创建矩形
            // 矩形参数计算（前面总结的4条公式）
            let top = Math.min(downPoint.y, upPoint.y)
            let left = Math.min(downPoint.x, upPoint.x)
            let width = Math.abs(downPoint.x - upPoint.x)
            let height = Math.abs(downPoint.y - upPoint.y)

            // 矩形对象
            const rect = new fabric.Rect({
                top,
                left,
                width,
                height,
                fill: 'transparent', // 填充色：透明
                stroke: '#000' // 边框颜色：黑色
            })

            // 将矩形添加到画布上
            canvas.add(rect)

            // 创建完矩形，清空 downPoint 和 upPoint。当然，你也可以不做这步。
            console.log("down:")
            console.log(this.downPoint)
            console.log("up:")
            console.log(this.upPoint)
            this.upPrecord=this.upPoint
            this.downPrecord=this.downPoint
            this.downPoint = null
            this.upPoint = null
        },
        canvasMouseDown(e) {
        // 鼠标左键按下时，将当前坐标 赋值给 downPoint。{x: xxx, y: xxx} 的格式
            this.downPoint = e.absolutePointer
        },
        canvasMouseUp(e) {
        // 鼠标左键按下时，将当前坐标 赋值给 downPoint。{x: xxx, y: xxx} 的格式
            this.upPoint = e.absolutePointer
            this.createRect()
            this.submitRegion()
        },
        createRectNew() {
            // 如果点击和松开鼠标，都是在同一个坐标点，不会生成矩形
            if (JSON.stringify(this.downPointNew) === JSON.stringify(this.upPointNew)) {
                return
            }
            console.log("I'm called")
            console.log(this.upPrecord)
            let canvas=this.canvas2
            let downPoint=this.downPointNew
            let upPoint=this.upPointNew
            // 创建矩形
            // 矩形参数计算（前面总结的4条公式）
            let top = Math.min(downPoint.y, upPoint.y)
            let left = Math.min(downPoint.x, upPoint.x)
            let width = Math.abs(downPoint.x - upPoint.x)
            let height = Math.abs(downPoint.y - upPoint.y)

            // 矩形对象
            const rect = new fabric.Rect({
                top,
                left,
                width,
                height,
                fill: 'transparent', // 填充色：透明
                stroke: '#000' // 边框颜色：黑色
            })

            // 将矩形添加到画布上
            canvas.add(rect)

            // 创建完矩形，清空 downPoint 和 upPoint。当然，你也可以不做这步。
            console.log("down:")
            console.log(this.downPointNew)
            console.log("up:")
            console.log(this.upPointNew)
            this.upPrecordNew=this.upPointNew
            this.downPrecordNew=this.downPointNew
            this.downPointNew = null
            this.upPointNew = null
        },
        canvasMouseDownNew(e) {
        // 鼠标左键按下时，将当前坐标 赋值给 downPoint。{x: xxx, y: xxx} 的格式
            this.downPointNew = e.absolutePointer
        },
        canvasMouseUpNew(e) {
        // 鼠标左键按下时，将当前坐标 赋值给 downPoint。{x: xxx, y: xxx} 的格式
            this.upPointNew = e.absolutePointer
            this.createRectNew()
            this.submitRegionNew()
        },
        addIntent(){
            if (this.add_text!='')
            {
                let int_div=d3.select("#input_list").append("div").style("display","inline-block").style("margin-left","30px")
                int_div.append("span").text(this.logic)
                int_div.append("span").text(this.add_text).style("margin-left","10px")
                this.extra_intents.push(this.add_text)
                this.extra_intent_logic.push(this.logic)
                this.extra_intent_modal.push("text")
            }
            else
            {
                let int_div=d3.select("#input_list").append("div").style("display","inline-block").style("margin-left","30px")
                int_div.append("span").text(this.logic)
                int_div.append("img").attr("src", this.segImPathNewTemp).attr("width","40").attr("height","40").style("margin-left","10px")
                this.extra_intents.push(this.segImPathNew)
                this.extra_intent_logic.push(this.logic)
                this.extra_intent_modal.push("image")
            }
        },
        addCompose(){
            d3.selectAll(".compose").style("display","inline-block")
            // d3.select("#modal").style("height","740px")
            let modal = document.getElementById('modal')
            modal.style.height = 600
        },
        closeMore() {
            d3.selectAll(".compose").style("display","none")
        },
        closeModal() {
            let modal = document.getElementById('modal')
            modal.style.display = 'none'
            let geIm=document.getElementById('InstructImage')
            geIm.style.display = 'none'
        },
        closeModal1() {
            let modal = document.getElementById('modal1')
            modal.style.display = 'none'
            document.getElementById("genImage").style.display="none"
        },
        // brushended({ selection }) {
        //     let newImgSet = []
        //     let imgP = []
        //     let bestOff = []
        //     let priceNums = []
        //     let bins=this.bins
        //     console.log(bins)
        //     let x=this.x
        //     for (let i in bins) {
        //         if ((x(bins[i].x0) >= selection[0] && x(bins[i].x1) <= selection[1])) {
        //             // console.log("#pr" + i);
        //             d3.select("#pr" + i).style('fill', 'red')

        //             // console.log("this is this")
        //             // console.log("this is this")
        //             // console.log(vm.Allcount)
        //             for (let u = 0; u < this.Allcount; u++) {
        //                 //  console.log(vm.allprices[u])
        //                 //  console.log(bins[i].x0)
        //                 if (this.allprices[u]>bins[i].x0 && this.allprices[u]<=bins[i].x1) {
        //                     newImgSet.push(this.allImgSet[u])
        //                 }
        //             }
        //             this.imgSet = newImgSet
        //             console.log(this.imgSet.length)
        //             for (let u = 0; u < this.imgSet.length; u++) {
        //                 let tp;
        //                 let s = this.imgSet[u]//[0]
        //                 let imp = s.split("#")[0]
        //                 let price = s.split("#")[1]
        //                 if (imp.charAt(0) != "s") { tp = imp.slice(1); }
        //                 else {
        //                     tp = "/" + imp;
        //                 }
        //                 let tend = '../assets' + tp;
        //                 imgP.push(tend);
        //                 bestOff.push(price)
        //                 priceNums.push(Number(price.split(' ')[0]))
        //             }

        //         }
        //         else {
        //             d3.select("#pr" + i).style('fill', '#D2D2D2')

        //         }
        //     }
        //     if (imgP.length>0)
        //     {
        //         this.imgPath = imgP;
        //         this.bestOffers = bestOff;
        //         this.priceNumbers = priceNums;
        //     }
        // },
        selectKey(event) {
            // console.log(event.target.innerText)
            let newImgSet = []
            let imgP = []
            let bestOff = []
            let priceNums = []
            let keytext = event.target.innerText
            // console.log(keytext.split(" ")[0])
            // console.log(this.imgTag)
            for (let i = 0; i < this.Allcount; i++) {
                let keyn=event.target.innerText.split("(")[0]
                let keylen=keyn.length
                let keyna=keyn.substring(0,keylen-1)
                if (this.imgTag[i] == keyna.toLowerCase() || this.imgTag[i] == keyna) {
                    // if (this.currentkey.indexOf(this.imgTag[i])==-1)
                    // {
                    //     this.currentkey.push(this.imgTag[i])
                    // }
                    newImgSet.push(this.allImgSet[i])
                }
            }
            this.imgSet = newImgSet
            // console.log(this.imgSet.length)
            for (let i = 0; i < this.imgSet.length; i++) {
                let tp;
                let s = this.imgSet[i]//[0]
                let imp = s.split("#")[0]
                let price = s.split("#")[1]
                if (imp.charAt(0) != "s") { tp = imp.slice(1); }
                else {
                    tp = "/" + imp;
                }
                let tend = '../assets' + tp;
                imgP.push(tend);
                bestOff.push(price)
                priceNums.push(Number(price.split(' ')[0]))
            }
            this.imgPath = imgP;
            this.bestOffers = bestOff;
            this.priceNumbers = priceNums;
        },
        selectCol(event) {
            // console.log(event.target.innerText)
            let coltext = event.target.innerText
            // console.log(coltext.split(" ")[0])
            let newImgSet = []
            let imgP = []
            let bestOff = []
            let priceNums = []
            // console.log(this.imgTag)
            for (let i = 0; i < this.Allcount; i++) {
                let coln=event.target.innerText.split("(")[0]
                let colen=coln.length
                let colna=coln.substring(0,colen-1)
                if (this.col_Tag[i] == colna) {
                    newImgSet.push(this.allImgSet[i])
                }
            }
            this.imgSet = newImgSet
            // console.log(this.imgSet.length)
            for (let i = 0; i < this.imgSet.length; i++) {
                let tp;
                let s = this.imgSet[i]//[0]
                let imp = s.split("#")[0]
                let price = s.split("#")[1]
                if (imp.charAt(0) != "s") { tp = imp.slice(1); }
                else {
                    tp = "/" + imp;
                }
                let tend = '../assets' + tp;
                imgP.push(tend);
                bestOff.push(price)
                priceNums.push(Number(price.split(' ')[0]))
            }
            this.imgPath = imgP;
            this.bestOffers = bestOff;
            this.priceNumbers = priceNums;
        },
        Revise(event) {
            // console.log(event.target)
            let formData = new FormData();
            let textContent = '|'//= document.getElementById('annotation').value
            for (let i = 0; i < 30; i++) {
                let label;

                let check = document.getElementById("anno" + i.toString())
                // console.log(check.checked)
                if (check.checked == true) {
                    textContent = textContent + "1" + ":" + document.getElementById("image" + i.toString()).src + "|"
                }
                else {
                    textContent = textContent + "0" + ":" + document.getElementById("image" + i.toString()).src + "|"
                }
            }
            formData.append("text", textContent);
            reviseText(formData, res => {
                // console.log(res);
            });

        },

    },
    components: { Main, Navbar, LeftMenu, SearchResult }
};

</script>
<style>
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


.image {
    width: 100%;
    display: block;
}

#selectedImage {
    width: 330px;
    height: 330px;
    margin-left: 20px;
    /* display: block; */
}

.modal-cover {
    width: 100%;
    height: 100%;
    position: fixed;
    top: 0;
    left: 0;
    background-color: rgba(0, 0, 0, .3);
    z-index: 99;
    display: none;
}

.modal {
    width: 40%;
    height: 440px;
    background-color: #fff;
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    border-radius: 10px;
}

.modal-cover1 {
    width: 100%;
    height: 100%;
    position: fixed;
    top: 0;
    left: 0;
    background-color: rgba(0, 0, 0, .3);
    z-index: 99;
    display: none;
}

.modal1 {
    width: 20%;
    height: 530px;
    background-color: #fff;
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    border-radius: 10px;
}

.el-button--keyword:active {

    background-color: #626aef;

}

.el-button--keyword:focus {

    background-color: #626aef;
}

.el-button--collection:active {

    background-color: #33A2E4;
}

.el-button--collection:focus {

    background-color: #33A2E4;
}

.detailTitle {
    font-weight: bold;
    font-size: 23px
}

.detailCon {

    font-size: 17px
}

.selection {
    fill: rgba(0, 0, 0, 0.3);
}</style>
