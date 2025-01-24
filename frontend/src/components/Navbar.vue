<template>
    <div style="font-family: Quicksand;">
        <!-- {{ message }} -->
        <!-- <el-menu class="el-menu-demo" mode="horizontal" :ellipsis="false">
            <el-menu-item index="0"> <svg xmlns="http://www.w3.org/2000/svg"
                    class="icon icon-tabler icon-tabler-brand-docker" width="50" height="56" viewBox="0 0 24 24"
                    stroke-width="1" stroke="black" fill="none" stroke-linecap="round" stroke-linejoin="round">
                    <path stroke="none" d="M0 0h24v24H0z" fill="none" />
                    <path
                        d="M22 12.54c-1.804 -.345 -2.701 -1.08 -3.523 -2.94c-.487 .696 -1.102 1.568 -.92 2.4c.028 .238 -.32 1.002 -.557 1h-14c0 5.208 3.164 7 6.196 7c4.124 .022 7.828 -1.376 9.854 -5c1.146 -.101 2.296 -1.505 2.95 -2.46z" />
                    <path d="M5 10h3v3h-3z" />
                    <path d="M8 10h3v3h-3z" />
                    <path d="M11 10h3v3h-3z" />
                    <path d="M8 7h3v3h-3z" />
                    <path d="M11 7h3v3h-3z" />
                    <path d="M11 4h3v3h-3z" />
                    <path d="M4.571 18c1.5 0 2.047 -.074 2.958 -.78" />
                    <line x1="10" y1="16" x2="10" y2="16.01" />
                </svg></el-menu-item>
            <div class="flex-grow" />
            <el-menu-item index="1">Processing Center</el-menu-item>
            <el-sub-menu index="2">
                <template #title>Workspace</template>
                <el-menu-item index="2-1">item one</el-menu-item>
                <el-menu-item index="2-2">item two</el-menu-item>
                <el-menu-item index="2-3">item three</el-menu-item>
                <el-sub-menu index="2-4">
                    <template #title>item four</template>
                    <el-menu-item index="2-4-1">item one</el-menu-item>
                    <el-menu-item index="2-4-2">item two</el-menu-item>
                    <el-menu-item index="2-4-3">item three</el-menu-item>
                </el-sub-menu>
            </el-sub-menu>
        </el-menu> -->

        <el-row :gutter="0" style="margin-top: 30px; margin-bottom: 30px;">
            <el-col :span="2">
                <div class="grid-content ep-bg-purple" style="position:absolute;top:2px;left:55px">

                    <svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-brand-docker" width="80"
                        height="86" viewBox="0 0 24 24" stroke-width="1" stroke="black" fill="none" stroke-linecap="round"
                        stroke-linejoin="round" style="margin-top:0px">
                        <path stroke="none" d="M0 0h24v24H0z" fill="none" />
                        <path
                            d="M22 12.54c-1.804 -.345 -2.701 -1.08 -3.523 -2.94c-.487 .696 -1.102 1.568 -.92 2.4c.028 .238 -.32 1.002 -.557 1h-14c0 5.208 3.164 7 6.196 7c4.124 .022 7.828 -1.376 9.854 -5c1.146 -.101 2.296 -1.505 2.95 -2.46z" />
                        <path d="M5 10h3v3h-3z" />
                        <path d="M8 10h3v3h-3z" />
                        <path d="M11 10h3v3h-3z" />
                        <path d="M8 7h3v3h-3z" />
                        <path d="M11 7h3v3h-3z" />
                        <path d="M11 4h3v3h-3z" />
                        <path d="M4.571 18c1.5 0 2.047 -.074 2.958 -.78" />
                        <line x1="10" y1="16" x2="10" y2="16.01" />
                    </svg>

                </div>
            </el-col>
            <el-col :span="12">
                <div class="grid-content ep-bg-purple" style="margin-top:25px">

                    <el-form :model="formData">
                        <el-form-item label="KapokNFT">
                            <!-- <el-input v-model="form.name" :input-style="{
                'margin-left': '10px',
                'margin-right': '10px',
            }"/> -->
                            <el-input v-model="formData.name" placeholder="textq" class="input-with-select">
                                <!-- <template #append>
                    <el-select v-model="select" placeholder="Select" style="width: 115px">
                        <el-option label="Restaurant" value="1" />
                        <el-option label="Order No." value="2" />
                        <el-option label="Tel" value="3" />
                    </el-select>
                </template> -->
                                <template #append>

                                    <div style="width: 50px; margin-top: -10px;">
                                        <el-button @click="showMoreSearch()">
                                            <el-icon :size="25" style="vertical-align: middle">
                                                <svg viewBox="0 0 1024 1024" xmlns="http://www.w3.org/2000/svg"
                                                    data-v-029747aa="" :transform="translate(0, 0, -90 * moreSearch)">
                                                    <path fill="currentColor"
                                                        d="M452.864 149.312a29.12 29.12 0 0 1 41.728.064L826.24 489.664a32 32 0 0 1 0 44.672L494.592 874.624a29.12 29.12 0 0 1-41.728 0 30.592 30.592 0 0 1 0-42.752L764.736 512 452.864 192a30.592 30.592 0 0 1 0-42.688zm-256 0a29.12 29.12 0 0 1 41.728.064L570.24 489.664a32 32 0 0 1 0 44.672L238.592 874.624a29.12 29.12 0 0 1-41.728 0 30.592 30.592 0 0 1 0-42.752L508.736 512 196.864 192a30.592 30.592 0 0 1 0-42.688z">
                                                    </path>
                                                </svg>
                                            </el-icon>
                                        </el-button>
                                    </div>
                                    <div style="width: 50px; margin-top: -10px;">
                                        <el-button @click="nftSearch()">
                                            <el-icon :size="25" style="vertical-align: middle">
                                                <Search />
                                            </el-icon>
                                        </el-button>
                                    </div>
                                    <div style="width: 50px; margin-top: -10px;">
                                        <el-button ref="img_button" @click="submitForm($event)">
                                            <el-icon :size="25" style="vertical-align: middle">
                                                <Camera />
                                            </el-icon>
                                        </el-button>

                                        <input ref="img_sub" type="file" name="img" id="pic_img" style="display: none;"
                                            @change="getFile($event)">
                                    </div>
                                    <div style="width: 50px; margin-top: -10px;">
                                        <el-button @click="showModal1($event)">
                                            <!-- <el-icon :size="25" style="vertical-align: middle">
                                                <Brush />
                                            </el-icon> -->
                                            ✨
                                        </el-button>
                                    </div>
                                    <!-- <el-button :icon="Search" />
                    <el-button :icon="Search" /> -->
                                </template>
                            </el-input>
                        </el-form-item>
                        <div v-show="moreSearch == -1 ? 0 : 1"
                            style="border: 1px solid rgb(220, 223, 230); padding: 20px; border-radius: 15px; width: calc(100% - 195px); font-weight: bold; margin-bottom: 30px; float: left;">
                            <div style="float: left; font-family: Quicksand;">Market Features</div>
                            <br>
                            <br>
                            <el-form-item label="Price">
                                <el-input v-model="formData.name" placeholder="the highest price, continuous rise"
                                    :input-style="{
                                        'float': 'right'
                                    }" />
                            </el-form-item>
                            <el-form-item label="Rarity">
                                <el-input v-model="formData.region" placeholder="rarest" />
                            </el-form-item>
                            <el-form-item label="Collection">
                                <el-input v-model="formData.type" placeholder="Doodles" />
                            </el-form-item>
                            <div style="float: left; font-family: Quicksand;">NFT Features</div>
                            <br>
                            <br>
                            <el-form-item label="Background Color">
                                <el-input v-model="formData.name" placeholder="red, blue, green" :input-style="{
                                    'float': 'right'
                                }" />
                            </el-form-item>
                            <el-form-item label="Style">
                                <el-input v-model="formData.region" placeholder="sketch, comic" />
                            </el-form-item>
                            <div style="float: left; font-family: Quicksand;">Personal Features
                                <el-button :style="{ 'border-radius': '30px', 'padding': '8px' }" @click="open()">
                                    <el-icon>
                                        <Plus />
                                    </el-icon>
                                </el-button>
                            </div>
                            <br>
                            <br>
                            <el-form-item v-for="(item, i) in addFeature" :key="'F' + i" :label="item">
                                <el-input v-model="formData.name" :input-style="{
                                    'float': 'right'
                                }" />
                            </el-form-item>
                        </div>
                        <div v-if="imgURL != ''"
                            style="border: 1px solid rgb(220, 223, 230); padding: 20px; border-radius: 15px; width: calc(190px); font-weight: bold; margin-bottom: 30px; float: right;">
                            <div style="float: left; font-family: Quicksand;">Selected Image</div>
                            <br>
                            <br>
                            <img :src="imgURL" alt="selected image" style="border: 1px solid rgb(220, 223, 230)"
                                width="150">
                        </div>
                    </el-form>
                </div>
            </el-col>
            <el-col :span="6">
                <div class="grid-content ep-bg-purple" />
            </el-col>
            <el-col :span="4">
                <el-menu class="el-menu-demo" mode="horizontal" :ellipsis="false">
                    <div class="flex-grow" />
                    <el-menu-item index="1" style="font-size: 22px;">Processing Center</el-menu-item>
                    <el-sub-menu index="2">
                        <template #title><span style="font-size: 22px;">Workspace</span></template>
                        <el-menu-item index="2-1">item one</el-menu-item>
                        <el-menu-item index="2-2">item two</el-menu-item>
                        <el-menu-item index="2-3">item three</el-menu-item>
                        <el-sub-menu index="2-4">
                            <template #title>item four</template>
                            <el-menu-item index="2-4-1">item one</el-menu-item>
                            <el-menu-item index="2-4-2">item two</el-menu-item>
                            <el-menu-item index="2-4-3">item three</el-menu-item>
                        </el-sub-menu>
                    </el-sub-menu>
                </el-menu>
            </el-col>
        </el-row>
        <el-row>
            <el-col span="16">
                <span style="margin-left: 520px;"></span>
                <!-- <el-button type="primary" plain @click="followup = followup * (-1)">
                    Ask a follow-up
                </el-button> -->

            </el-col>
            <el-col span="8">
                <el-form v-if="followup == 1">
                    <el-form-item label="">

                        <el-input v-model="formData1.name" placeholder="What else?" class="input-with-select">

                            <template #append>

                                <div style="width: 50px; margin-top: -10px;">
                                    <el-button @click="nftSearch()">
                                        <el-icon :size="25" style="vertical-align: middle">
                                            <Search />
                                        </el-icon>
                                    </el-button>
                                </div>


                            </template>
                        </el-input>
                    </el-form-item>
                </el-form>
            </el-col>
        </el-row>
    </div>
</template>
<script>
import { ElMessage, ElMessageBox } from 'element-plus'
import { useDataStore } from "../stores/counter";
import { postImg, postText } from '../service/module/dataService'

export default {
    name: 'navbar',
    props: ['textQuery', 'message'],
    // props: {
    //     textQuery: {
    //         type: String,
    //         required: true
    //     },
    //     message: {
    //         type: Array,
    //         required: true
    //     },
    // }, 

    watch: {
        textQuery: {
            immediate: true,
            handler(value) {
                this.textq = value
            }
        },
        //   message: {
        //     immediate: true,
        //     handler(nvalue) {
        //       console.log("fkkkkkkkk")
        //       this.tag_list=nvalue
        //     }
        //   }
        message(nvalue) {
            console.log("fkkkkkkkk", nvalue)

            this.tag_list = nvalue
        }
    },
    data() {
        return {
            moreSearch: -1,
            formData: {
                name: ''
            },
            formData1: {
                name: ''
            },
            imgData: {
                accept: "image/png, image/jpeg, image/jpg"
            },
            addFeature: [],
            file: '',
            imgURL: '',
            followup: -1,
            textq: "",
            tag_list: [],

        }
    },
    // mounted() {
    //     this.formData.name=this.textQuery
    // },
    methods: {
        translate(x, y, r) {
            return `translate(${x}, ${y}) rotate(${r})`;
        },
        jump() {
            // const router = useRouter();
            this.$emit("fetchData")
            console.log("hello")
            // this.$router.push({ path: 'result' });
            // this.$emit("fetchData")
        },
        // nftSearch() {
        //     this.jump();
        // },
        nftSearch() {
            const dataStore = useDataStore();
            // Image
            if (this.file != '' ) {
                let formData = new FormData();
                formData.append('image', this.file);
                postImg(formData, res => {
                    console.log(res);
                    this.jump();
                });
            }
            // Text
            else if (this.file == '' && this.formData.name != '') {
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
                    this.jump();
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
        getFile(event) {
            console.log(event.target.files);
            this.file = event.target.files[0];
            console.log(this.file);


            event.preventDefault();
            let type = this.file.type;
            let size = this.file.size;
            // if (this.imgData.accept.indexOf(type) == -1) {
            //     ElMessage({
            //         type: 'error',
            //         message: 'Invalid Image Format',
            //     })
            // }
            let URL = window.URL || window.webkitURL;
            let imgURL = URL.createObjectURL(this.file);
            this.imgURL = imgURL;
        },
        submitForm(event) {
            this.$refs.img_sub.dispatchEvent(new MouseEvent('click'))

        },
        open() {
            const vm = this;
            ElMessageBox.prompt('Please add a feature', 'Feature Adding', {
                confirmButtonText: 'Add',
                cancelButtonText: 'Cancel',
                inputPattern: ''
            })
                .then(({ value }) => {
                    vm.addNFTFeature(value);
                    ElMessage({
                        type: 'success',
                        message: `Successfully added feature: ${value}`,
                    })
                })
        },
        showMoreSearch() {
            this.moreSearch = -this.moreSearch;
            // console.log(this.moreSearch)
        },
        showModal1(event) {
            let modal = document.getElementById('modal1')
            modal.style.display = 'block'
        },
        addNFTFeature(featureName) {
            this.addFeature.push(featureName);
        }
    },
    // created() {
    // },
    created() {
        this.tag_list = this.message
    },
    mounted() {
        //     console.log("#############")
        // console.log(this.$route)

        // console.log("#############")
    },
}
</script>
<style scoped>
.flex-grow {
    flex-grow: 1;
}

.el-input {
    --el-input-border-radius: 25px;
    height: 35px;
    font-size: 15px;
    font-family: Quicksand, system-ui;
    color: black
        /* padding-right: 15%; */
}

/* .el-form-item {
    font-size: 80px;
} */
.el-form {
    --el-form-label-font-size: 50px;
}

.el-input {
    font-size: 22px;
}
</style>
