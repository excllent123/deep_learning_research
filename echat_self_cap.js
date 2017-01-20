// Schema:
// date,AQIindex,PM2.5,PM10,CO,NO2,SO2
var dataBJ = [
    [55,9,56,0.46,18,6,1],
   
    [46,5,49,0.28,10,6,31]
];

var dataGZ = [
  
    [93,77,104,1.165,53,7,23],
    [99,130,227,3.97,55,15,24],
    [146,84,139,1.094,40,17,25],
    [113,108,137,1.481,48,15,26],
    [81,48,62,1.619,26,3,27],
    [56,48,68,1.336,37,9,28],
    [82,92,174,3.29,3,13,29],
    [106,116,188,3.628, 50,16,30],
    [118,50,0,1.383, 66,11,31]
];

var dataSH = [
 
];

var lineStyle = {
    normal: {
        width: 1,
        opacity: 0.5
    }
};

option = {
    backgroundColor: '#161627',
    title: {
        text: '能力雷達圖',
        left: 'center',
        textStyle: {
            color: '#eee'
        }
    },
    legend: {
        bottom: 5,
        data: ['個人能力'],
        itemGap: 20,
        textStyle: {
            color: '#fff',
            fontSize: 14
        },
        selectedMode: 'single'
    },
    // visualMap: {
    //     show: true,
    //     min: 0,
    //     max: 20,
    //     dimension: 6,
    //     inRange: {
    //         colorLightness: [0.5, 0.8]
    //     }
    // },
    radar: {
        indicator: [
            {name: '人工智能', max: 160},
            {name: '數據分析', max: 150},
            {name: '數學能力', max: 300},
            {name: '工業知識', max: 6},
            {name: '業務能力', max: 100},
            {name: '個人品格', max: 20}
        ],
        shape: 'circle',
        splitNumber: 5,
        name: {
            textStyle: {
                color: 'rgb(238, 197, 102)'
            }
        },
        splitLine: {
            lineStyle: {
                color: [
                    'rgba(238, 197, 102, 0.1)', 'rgba(238, 197, 102, 0.2)',
                    'rgba(238, 197, 102, 0.4)', 'rgba(238, 197, 102, 0.6)',
                    'rgba(238, 197, 102, 0.8)', 'rgba(238, 197, 102, 1)'
                ].reverse()
            }
        },
        splitArea: {
            show: false
        },
        axisLine: {
            lineStyle: {
                color: 'rgba(238, 197, 102, 0.5)'
            }
        }
    },
    series: [
 
        {
            name: '個人能力',
            type: 'radar',
            lineStyle: lineStyle,
            data: dataGZ,
            symbol: 'none',
            itemStyle: {
                normal: {
                    color: 'rgb(238, 197, 102)'
                }
            },
            areaStyle: {
                normal: {
                    opacity: 0.05
                }
            }
        }
    ]
};