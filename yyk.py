import streamlit as st
import pandas as pd

#  全局页面配置
st.set_page_config(
    page_title="我的实训作品集",
    page_icon="📚",
    layout="wide"
)

# ---------------------- 修正选项卡逻辑 ----------------------
menu_items = [
    "首页",
    "实训1：音乐歌单日志",
    "实训2：南宁美食流量数据",
    "实训4：相册展示",
    "实训5：简易音乐播放器",
    "实训6：仙逆视频播放"
]

# 创建选项卡，并直接在对应选项卡中渲染内容（无需单独定义current_page）
tabs = st.tabs(menu_items)

# 1. 首页内容（直接放在第一个选项卡中）
with tabs[0]:
    st.title("📁 我的Streamlit实训作品集")
    st.markdown("---")
    st.image("https://n.sinaimg.cn/sinacn20114/277/w1597h1080/20190206/07c0-hsqyiwt7507967.jpg", use_container_width=True)
    st.markdown("""
    ### 👋 实训合集简介
    基于Streamlit开发的Python实战项目，包含5个核心模块：
    - 音乐歌单数据可视化
    - 南宁美食流量分析
    - 交互式相册
    - 简易音乐播放器
    - 视频剧集播放
    
    ### 📌 核心目标
    掌握Streamlit交互组件、数据可视化、多媒体展示能力
    """)
    st.markdown("---")


# 2. 实训1：音乐歌单日志（放在第二个选项卡中）
with tabs[1]:
    st.title("音乐--薄荷鱼🎧的歌单日志")
    st.markdown('***')
    st.header('基础听歌数据🎵')
    st.markdown('总收藏歌曲：:red[328首]')
    st.markdown('年度听歌时长：:red[1280小时]')
    st.markdown('最近一周听歌：:red[28小时]')
    st.markdown('常用听歌设备：:red[手机/耳机🎧️]')

    st.markdown('### 分类列表📝')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('**最近**')
        st.markdown('### 95%')
        st.markdown(':green[↑+2%]')
    with col2:
        st.markdown('**收藏**')
        st.markdown('### 87%')
        st.markdown(':red[↓-1%]')
    with col3:
        st.markdown('**博客**')
        st.markdown('### 68%')
        st.markdown(':red[↓-10%]')

    st.markdown('***')
    st.markdown('### 歌单听歌进度⏯︎')
    st.markdown('「通勤必备」歌单完成度')
    progress = 0.65
    st.progress(progress)

    st.markdown('***')
    st.markdown('### 歌单听歌日志✏️')
    log_data = {
        "日期": ["2025-12-01", "2025-12-05", "2025-12-10"],
        "歌单名称": ["《小孩》", "《后陡门的夏》", "《宠爱》"],
        "状态": ["✅ 已听完", "🔄 收听中", "❌ 未开始"],
        "喜爱度": ["⭐️⭐️⭐️⭐️", "⭐️⭐️⭐️", "⭐️⭐️⭐️⭐️⭐️"]
    }
    st.dataframe(log_data, hide_index=True, use_container_width=True)

    st.markdown('***')
    st.markdown('### 🎵 音乐收藏脚本')
    music_script = '''def collect_favorite_songs():
    favorite_list = []
    while True:
        song = input("输入想收藏的歌曲：")
        if song == "结束":
            print("收藏完成！当前歌单：", favorite_list)
            return favorite_list
        else:
            favorite_list.append(song)
            print(f"已收藏：{song}")'''
    st.code(music_script, language="python", line_numbers=True)

    st.markdown('***')
    st.markdown('### 🎧 歌单更新提示')
    st.markdown('> :green[>> SYSTEM MESSAGE: 新歌曲已加入歌单...]')
    st.markdown('> :blue[>> NEW SONG: 《七里香》（周杰伦）]')
    st.markdown('> :orange[>> UPDATE TIME: 2025-12-12 19:30:00]')
    st.markdown('')
    st.markdown('歌单状态：已更新 | 播放状态：可收听')


# 3. 实训2：南宁美食流量数据（放在第三个选项卡中）
with tabs[2]:
    st.header('南宁美食流量数据🍲')
    st.markdown('***')  
    data = {
        '朴大叔拌饭':[200, 150, 180,400, 150, 280,210, 150, 190,200, 150, 180],
        '大叔的虾':[120, 160, 123,500, 180, 380,270, 190, 180,400, 180, 280],
        '喜虾客':[110, 100, 160,300, 170, 280,230, 160, 380,200, 150, 180],
        '成都冒烤鸭':[200, 150, 180,200, 150, 480,260, 250, 160,300, 190, 380],
        '二三麻辣烫':[120, 160, 123,100, 190, 180,250, 140, 300,100, 100, 180]
    }
    month_index = pd.Series(['01月','02月','03月','04月','05月','06月','07月','08月','09月','10月','11月','12月'], name='月份')
    flow_df = pd.DataFrame(data, index=month_index)

    st.header('📍南宁美食地图')
    map_data={
         'latitude':[22.856562,22.835680,22.830934,22.843907,22.789793],
         'longitude':[108.244514,108.244171,108.218079,108.291893,108.312836]
        }
    mp_df=pd.DataFrame(map_data) 
    st.map(mp_df)

    st.markdown('***')
    st.header('⭐ 餐厅评分')
    pingfen_data = {
        "餐厅名称": ["朴大叔拌饭", "大叔的虾", "喜虾客", "成都冒烤鸭", "二三麻辣烫"],
        "评分": [4, 5, 4.5, 4.5, 4.8]  
    }
    score_df = pd.DataFrame(pingfen_data).set_index("餐厅名称")
    st.bar_chart(score_df, color="#1f77b4")

    st.markdown('***')
    st.header('美食流量表格数据📖')
    st.dataframe(flow_df, use_container_width=True)

    st.markdown('***')
    st.header('美食流量折线图📉')
    st.line_chart(flow_df)

    st.markdown('***')
    peak_data = {
        "时段": ["11.0", "11.5", "12.0", "12.5", "13.0", "13.5", "14.0", "14.5", "15.0", "15.5", "16.0", "16.5", "17.0", "17.5", "18.0", "18.5", "19.0"],
        "朴大叔拌饭": [40, 80, 80, 60, 50, 50, 45, 45, 40, 40, 45, 70, 80, 80, 75, 70, 60],
        "大叔的虾": [40, 95, 85, 70, 65, 60, 55, 50, 45, 40, 40, 50, 60, 75, 85, 80, 70],
        "喜虾客": [40, 85, 70, 55, 50, 45, 40, 80, 35, 35, 40, 65, 85, 80, 75, 70, 65],
        "成都冒烤鸭": [40, 95, 85, 70, 65, 60, 55, 50, 45, 40, 40, 66, 60, 75, 85, 80, 70],
        "二三麻辣烫": [40, 85, 70, 55, 50, 45, 55, 40, 77, 35, 40, 65, 85, 86, 75, 70, 65]
    }
    peak_df = pd.DataFrame(peak_data).set_index("时段")
    st.header('⏱️ 用餐高峰时段')
    st.area_chart(peak_df, color=["#2E86AB", "#A23B72", "#F18F01","#3498DB", "#2ECC71"])


# 4. 实训4：相册展示（放在第四个选项卡中）
with tabs[3]:
    st.title("我的相册📷")
    if 'ind_album' not in st.session_state:
        st.session_state['ind_album'] = 0
    images = [
        {
            'url': "https://img95.699pic.com/photo/40243/6994.jpg_wh860.jpg",
            'text': '草莓蛋糕'
        },
        {
            'url': "http://wdmcake.cn/images/upload/Image/750_576.jpg",
            'text': '芒果蛋糕'
        },
        {
            'url': "https://img95.699pic.com/photo/60058/4401.jpg_wh860.jpg",
            'text': '巧克力蛋糕'
        }
    ]
    st.image(
        images[st.session_state['ind_album']]['url'],
        caption=images[st.session_state['ind_album']]['text'],
        use_container_width=True
    )
    def nextImg():
        st.session_state['ind_album'] = (st.session_state['ind_album'] + 1) % len(images)
    def fanhuiImg():
        st.session_state['ind_album'] = (st.session_state['ind_album'] - 1) % len(images)
    c1, c2 = st.columns(2)
    with c1:
        st.button("上一张", on_click=fanhuiImg, use_container_width=True)
    with c2:
        st.button("下一张", on_click=nextImg, use_container_width=True)


# 5. 实训5：简易音乐播放器（放在第五个选项卡中）
with tabs[4]:
    st.title("🎵简易音乐播放器")
    st.text("使用streamlit制作的简单音乐播放器，支持切歌和基本播放控制")
    if 'ind_music' not in st.session_state:
        st.session_state['ind_music'] = 0
    music_data = [
        {
            "name": "再见（good bye）",
            "singer": "G.E.M.邓紫棋",
            "url": "http://p2.music.126.net/kVwk6b8Qdya8oDyGDcyAVA==/1364493930777368.jpg",
            "time": "5:55",
            "map3": "https://music.163.com/song/media/outer/url?id=2709447832.mp3",
            "text": "专辑封面"
        },
        {
            "name": "最好的我们",
            "singer": "陈飞宇",
            "url": "https://so1.360tres.com/t0172e1255c71f0292e.jpg",
            "time": "4:30",
            "map3": "https://music.163.com/song/media/outer/url?id=2709447832.mp3",
            "text": "专辑封面"
        },
        {
            "name": "起风了",
            "singer": "买辣椒也用券",
            "url": "https://p1.music.126.net/diGAyEmpymX8G7JcnElncQ==/109951163699673355.jpg?param=300y300",
            "time": "3:45",
            "map3": "https://music.163.com/song/media/outer/url?id=1330348068.mp3",
            "text": "专辑封面"
        }
    ]
    idx = st.session_state['ind_music']
    cur = music_data[idx]
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.image(music_data[idx]['url'], use_container_width=True)
        st.caption("专辑封面")
    with col_right:
        st.subheader(cur['name'])
        st.write(f"歌手: {cur['singer']}")
        st.write(f"时长: {cur['time']}")
        btn1, btn2 = st.columns(2)
        with btn1:
            st.button(
                "◀上一首",
                on_click=lambda: st.session_state.update(ind_music=(st.session_state['ind_music']-1)%len(music_data)),
                use_container_width=True
            )
        with btn2:
            st.button(
                "下一首 ▶",
                on_click=lambda: st.session_state.update(ind_music=(st.session_state['ind_music']+1)%len(music_data)),
                use_container_width=True
            )
    st.audio(cur['map3'], format="audio/mpeg")


# 6. 实训6：仙逆视频播放（放在第六个选项卡中）
with tabs[5]:
    st.title('仙逆')
    st.markdown(':yellow[🔥18564 · 内地 · 2023 · 东方玄幻 · 东方仙侠 · 玄幻修真]')
    st.markdown(':yellow[豆瓣高分]')
    st.markdown(':green[更新至119集 · 全128集]')
    st.markdown('***')
    st.markdown('#### 📖简介')
    st.text('改编自耳根同名小说《仙逆》，讲述了乡村平凡少年王林以心中之感动，逆仙而修，求的不仅是长生，更多的是摆脱那背后的蝼蚁之身。他坚信道在人为，以平庸的资质踏入修真仙途，历经坎坷风雨，凭着其聪睿的心智，一步一步走向巅峰，凭一己之力，扬名修真界。')
    st.markdown('***')
    st.markdown('#### 📖主角介绍：')
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(
            "https://ts3.tc.mm.bing.net/th/id/OIP-C._Nh_CciYAIxiiLvmPpwyCwHaLf?cb=ucfimg2&ucfimg=1&rs=1&pid=ImgDetMain&o=7&rm=3",
            caption="王林（主角）",
            width=200
        )
    with c2:
        st.markdown("""
        ### 王林
        - **身份**：《仙逆》主角，原为赵国青牛镇平凡少年，后成为修真界顶级强者
        - **性格**：心智坚韧、杀伐果断，重情重义但不圣母，信奉“人不犯我我不犯人”
        - **经历**：因资质平庸被恒岳派外门收留，意外获“天逆珠”后开启逆袭之路，历经朱雀星、界内界外等多界征战，最终踏破轮回，成就“逆尘界”之主
        - **核心特质**：以“逆”为道，打破命运枷锁，从蝼蚁之身逆袭为掌控自身命运的强者
        """)
    st.markdown('***')
    video_arr = [
        {
            'url': 'https://apd-1f8d1cf95c10c976e64a0cede2e13cda.v.smtcdns.com/vhot2.qqvideo.tc.qq.com/A-0eq8He4Nlgp0f8Xh3i1WhfsP2x9Edc3uMfD0-F8DNQ/B_3k--xdVBUHYl1q0K2jODe5Kf15PJzXYk2N-VxUeXXOHixTUVK0pqaE3wv4HhTRC4Mq3OuO2Yb97QDz7HGmGz-9Es4IliRhjqoPY2uqrs-JU0cqxafmT-MvEkl4o_7uNVKIvFvNEedTY97p2xGmq39Q/svp_50069/gzc_1000035_0bc3yeapmaaa74aepsz6ijuzlqod63aqb5sa.f632.mp4?vkey=1EF4B07767D991832AE03DCE38D73C2A8307D8C95D11FC72C89345370976304EE33D8C56C398C5E4DE665531606EB33B9B0D1DC271C05F01D48092336FF3E34A8F6807C0F6FA3FBEA67D1CB399FC94FB47EF15DCA52A975D245FFC0796731CCF95F392A712075D08E3006BD723222A677C59EB93F5CA10872D4B24285362E07A',
            'title': '🎬第1集',
            'episode': 1
        },
        {
            'url': 'https://apd-c6dda2dc0a4a6b3b7cd60f4d47006a5a.v.smtcdns.com/vhot2.qqvideo.tc.qq.com/AUf1UgljygXot4dgnLQ1xKg-8f1hK1vFiynQ9iaUgifo/B_3k--xdVBUHYl1q0K2jODe93bgW8JJm4xNGc0UoJoinH69qvxFYF2M_vk0k7uW-XVV8HCUwToSbX-AFiQoxg33S4r3YXeQbPs9L-KovFY_LY0cqxafmT-MvEkl4o_7uNVKIvFvNEedTY97p2xGmq39Q/svp_50069/gzc_1000035_0bc3veac2aaa2yakt3z6l5uzlkodfwuqalka.f632.mp4?vkey=1626EFA3A3EBD9E199AFF5A9C348A021799F71ACB332908A288CA3F59CB1DA17A09A1B66B3C26926095C9AF1AD3254B5F108F4AB92EA704407595E40C5F789EC07A144BA3B1CCC1CDBD8042D438CB1B8DBD224808FEDC34228DA205FEB2BE5EABC35E2D389EF26FEB5FCB382C1DFCC3500278F27272BA0D1EE3AA94CB7066CAB',
            'title': '🎬第2集',
            'episode': 2
        },
        {
            'url': 'https://apd-915c25f830109f6649661d6469110d3c.v.smtcdns.com/vhot2.qqvideo.tc.qq.com/A2dgIN4DnChdWyiQ7C-XaIXqXApPIVv8XsgOoCW3LgEo/B_3k--xdVBUHYl1q0K2jODe32KcEJzV0zfhAFCg7hYjsjrD9TjRf3IbOD3RKyQe3PnQeGWtooH1HGcVCx1_iD8-2GD8IuGFlu3KU1IhJPr5Xc0cqxafmT-MvEkl4o_7uNVKIvFvNEedTY97p2xGmq39Q/svp_50069/gzc_1000035_0bc3vmaaqaaaieahokb6kfuzlk6dbcvqacca.f632.mp4?vkey=E604868C5735A0018AAE5FCD5EFF0C3027827F5A563E27F1D471C03C46F6A9837025DCF53BF4D4A5603D723CEF7BC419E94D156CB78F3A610D1CB4E0096C0464F753DC5521B9A967CF686D5E7C777DCF9273E93BDE1573172C53A1EBC00DBD65BB2ABBC6B189F7ABD86398E5A36A44E2FF93C1463F976953DAB636435F007508',
            'title': '第3集',
            'episode': 3
        },
        {
            'url': 'https://www.w3school.com.cn/example/html5/mov_bbb.mp4',
            'title': '🎬第4集',
            'episode': 4
        }
    ]
    if 'ind_video' not in st.session_state:
        st.session_state['ind_video'] = 0
    st.subheader(video_arr[st.session_state['ind_video']]['title'])
    st.video(video_arr[st.session_state['ind_video']]['url'], autoplay=True)
    def play(i):
        st.session_state['ind_video'] = int(i)
    cols = st.columns(len(video_arr))
    for i in range(len(video_arr)):
        with cols[i]:
            st.button(f'第{i+1}集', use_container_width=True, on_click=play, args=([i]))
