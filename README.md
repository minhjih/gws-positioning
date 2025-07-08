# gws-positioning

## Dataset 부연 설명
- 3* 30* 30 사이즈 CSI
- RP 총 24개
- RP 당 데이터 30번 수집
따라서 scenex_cir.npy는 총 (24* 30)* 3* 30* 30의 사이즈 [720, 3, 30, 30]

## Indoor Channel Generator
options:
  -h, --help     show this help message and exit
  --seed SEED    Random seed value (default: 43)
  --scene SCENE  Scene number (default: 0)