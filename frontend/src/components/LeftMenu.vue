<template>
    <div ref="leftMenu"
        style="float: left; padding-left: 0px; font-family: Quicksand; width: 100%;margin-bottom: 30px; border-right: 1px solid rgb(220, 223, 230);">
        <!-- <div style="width: 100%;">
            <div style="margin-bottom: 20px;">Sort by</div>
            <div style="width: 100%">
                <el-cascader v-model="sort_value" :options="sort_options" />
            </div>
            <el-divider />
        </div> -->
        <!-- <div style="width: 100%;">
            <div style="margin-bottom: 20px;">Generated NFT</div>
            <div style="width: 100%">
                <img src="https://shadow.elemecdn.com/app/element/hamburger.9cf7b091-55e9-11e9-a976-7f4d0b07eef6.png"
                    :width="elWidth">
            </div>
            <el-divider />
        </div> -->
        <div style="width: 100%;" v-show="select_img">
            <div style="margin-bottom: 20px;">Selected Image</div>
            <div style="width: 100%">
                <img :src="imgURL" :width="elWidth">
            </div>
            <el-divider />
        </div>
        <!-- <div style="width: 100%;">
            <div style="margin-bottom: 5px;">Price</div>
            <div style="width: 80%; margin-left: 10%;">
                <span style="float: left;">$1</span><span style="float: right;">$100</span>
                <el-slider v-model="value" range />
            </div>
            <el-divider />
        </div> -->
        <!-- <el-divider content-position="left">Collections</el-divider> -->
        <div style="width: 100%;">
            <div style="margin-bottom: 20px;font-size: large;background-color: rgb(245, 245, 245);">Collection</div>
            <div style="width: 100%; float: left;">
                <div v-for="(item, i) in collections" style="float: left; margin-left: 15px;">
                    <el-checkbox v-model="item.type" :label="item.id+'('+item.count+')'" size="large" />
                </div>
            </div>
        </div>
        <br>
        <!-- <el-divider content-position="left">Key Words</el-divider> -->
        <div style="width: 100%;">
            <div style="margin-bottom: 20px;font-size: large;background-color: rgb(245, 245, 245);margin-top: 334px;">Key Words</div>
            
            <div style="width: 100%; float: left;">
                <div v-for="(item, i) in collections" style="float: left; margin-left: 15px;">
                    <el-checkbox v-model="item.type" :label="item.id" size="large" />
                </div>
            </div>
        </div>
        <div id="chart" style="width:100%;height:400px;margin-top:300px;">
            <div style="margin-top: -50px;font-size: large;height:30px;background-color: rgb(245, 245, 245);">Price</div>
        </div>
    </div>
    <!-- <el-divider direction="vertical" ></el-divider> -->
</template>
<script>
import { useDataStore } from "../stores/counter";
export default {
    name: 'APP',
    props: [''],
    data() {
        return {
            select_img: 0,
            imgURL: '',
            elWidth: 0,
            value: [0, 100],
            sort_value: [],
            sort_options: [{
                value: 'price',
                label: 'Price',
                children: [
                    { value: 'up', label: 'Ascending order' },
                    { value: 'down', label: 'Descending order' }
                ]
            }, {
                value: 'rarity',
                label: 'Rarity',
                children: [
                    { value: 'up', label: 'Ascending order' },
                    { value: 'down', label: 'Descending order' }
                ]
            }],
            marks: {
                0: '$0',
                100: '$100'
            },
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
                {
                    id: 'BEANZ Official',
                    type: true,
                    count: 15
                },
                {
                    id: 'goblintown',
                    type: true,
                    count: 15
                },
                {
                    id: 'Karafuru',
                    type: true,
                    count: 15
                },
                {
                    id: 'Meebit',
                    type: true,
                    count: 15
                },
                {
                    id: 'mfers',
                    type: true,
                    count: 15
                },
                {
                    id: 'Moonbirds',
                    type: true,
                    count: 15
                },
                {
                    id: 'NFT Worlds',
                    type: true,
                    count: 15
                },
                {
                    id: 'PhantaBear',
                    type: true,
                    count: 15
                },
            ]
        }
    },
    methods: {
    },
    created() {
    },
    mounted() {
        this.elWidth = this.$refs.leftMenu.offsetWidth - 40;
        const dataStore = useDataStore();
        const img_file = dataStore.imgFile;
        if (img_file != '') {
            let URL = window.URL || window.webkitURL;
            let imgURL = URL.createObjectURL(img_file);
            this.imgURL = imgURL;
            this.select_img = !this.select_img;
            console.log(img_file);
        }
    },
}
</script>
<style scoped>

</style>
