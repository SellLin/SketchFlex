<template>
    <div>
        <el-row>
            <el-col v-for="(o, index) in 100" :key="o" :span="5" :offset="index % 4 == 0 ? 0 : 1"
                style="margin-bottom: 20px;">
                <el-card :body-style="{ padding: '0px' }">
                    <img src="https://shadow.elemecdn.com/app/element/hamburger.9cf7b091-55e9-11e9-a976-7f4d0b07eef6.png"
                        class="image" />
                    <div style="padding: 14px">
                        <span>Yummy hamburger</span>
                        <div class="bottom">
                            <el-button text class="button">Operating</el-button>
                        </div>
                    </div>
                </el-card>

            </el-col>
        </el-row>
    </div>
</template>
<script>
export default {
    name: 'APP',
    props: ['imgSet'],
    data() {
        return {

        }
    },
    methods: {
    },
    created() {
    },
    mounted() {
        console.log(this.imgSet);
    },
}
</script>
<style scoped>

.image {
    width: 100%;
    display: block;
}
</style>
