# Non-blind deconvolution methods benchmark

A benchmark for non-blind deconvolution methods: classical algorithms vs SOTA neural models.

---

## **Installation**

1. Install requirements (python >= 3.9):

```
make install
```

2. Download prepared data:

```
TODO
```

or

Download raw data:

```
TODO
```

and unpack it:

```
make prepare_raw_data
```

## **Validation**

Just simply run:
```
make test
```

---

## **Sources of data**

### Kernels:

1. Motion blur:

    1.1 Levin et al. Understanding and evaluating blind deconvolution algorithms. [Paper](https://ieeexplore.ieee.org/abstract/document/5206815), [data source](https://webee.technion.ac.il/people/anat.levin/). Total: 8 kernels.

    1.2 Sun et al. Edge-based Blur Kernel Estimation Using Patch Priors. [Paper & data source](https://cs.brown.edu/people/lbsun/deblur2013/deblur2013iccp.html). Total: 8 kernels (img_1_kernel{i}_OurNat5x5_kernel.png).

    1.3 Generated with simulator taken from [RGDN](https://github.com/donggong1/learn-optimizer-rgdn). Source code: `src/data/generate/motion_blur.py`. 


2. Eye PSF:

    2.1 Generated with our own simulator. Size: 256*256.

3. Gauss:

    3.1 Generated with [this script](https://github.com/birdievera/Anisotropic-Gaussian/blob/master/gaussian_filter.py).


### Ground truth images

1. [BSDS300](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/)

2. [Sun et al](https://cs.brown.edu/people/lbsun/deblur2013/deblur2013iccp.html). Images are taken from [here](https://drive.google.com/drive/folders/1Mb_mhtLG6N7CwiCMBnBMlJZyaqxQM-Nl).

3. Precompensation dataset [TBA]


---

## **Models and algorithms**

1. Wiener filter (as baseline): source code in `src/deconv/classic/wiener.py`;

2. [USRNet](https://github.com/cszn/USRNet): source code in `src/deconv/neural/usrnet`;

3. [DWDN](https://github.com/dongjxjx/dwdn): source code in `src/deconv/neural/dwdn`;

4. [KerUnc](https://github.com/ysnan/NBD_KerUnc): source code in `src/deconv/neural/kerunc`;

5. [RGDN](https://github.com/donggong1/learn-optimizer-rgdn): source code in `src/deconv/neural/rgdn`.


Example of each model inference can be found [here](notebooks/models.ipynb).

---

## **Tips**

### SQL

- If you work in VS Code, you can use this [extention for SQLLite](https://marketplace.visualstudio.com/items?itemName=alexcvzz.vscode-sqlite) to make your work easier.

- To calculate statistics (e.g. std and median), [this extention](https://github.com/nalgeon/sqlean/blob/main/docs/install.md) is used here. Just download precompiled binaries suitable for your OS and unpack them to a folder (`sqlean` in my case). That's it!

- SQL queries to analyze benchmarking results can be found [here](results/queries.sql).

### Running the code

- If old torch version (we use 1.7.1 since we took the source code for neural models as is) is not compatible with your CUDA version, you can run this code in Docker container. Instructures are below.

---


## **How to run docker container**

1. Build image:

```
make build
```

2. Run container:

```
make run
```

3. Execute inside the container:

```
make exec
```

4. Run inside the container:

```
make test
```


## Benchmarking results

It should be noted, that only *motion blur* is the type of blur common for all models, so the most correct comparison of models should be done by this domain.

<table><tr><th>blur_type</th><th>discretization</th><th>noised</th><th>model</th><th>MEDIAN(psnr)</th><th>AVG(psnr)</th><th>STD(psnr)</th><th>MEDIAN(ssim)</th><th>AVG(ssim)</th><th>STD(ssim)</th><tr><tr><td>eye_blur</td><td>float</td><td>0</td><td>dwdn</td><td>19.8090559476429</td><td>20.1494657639477</td><td>3.54284789516464</td><td>0.662738063203178</td><td>0.659623347397812</td><td>0.15206098915741</td></tr><tr><td>eye_blur</td><td>float</td><td>0</td><td>kerunc</td><td>21.431442384317</td><td>22.0005514300707</td><td>4.49194717120774</td><td>0.663858632914503</td><td>0.674683565729147</td><td>0.167940890820948</td></tr><tr><td>eye_blur</td><td>float</td><td>0</td><td>usrnet</td><td>29.7779501769173</td><td>30.6352585223495</td><td>5.19342618711741</td><td>0.906793448248148</td><td>0.891516731313286</td><td>0.0860722453367984</td></tr><tr><td>eye_blur</td><td>float</td><td>0</td><td>wiener_blind_noise</td><td>20.5436712741999</td><td>20.8100056768342</td><td>3.84777897290349</td><td>0.734622517044258</td><td>0.725412603298857</td><td>0.13766949530678</td></tr><tr><td>eye_blur</td><td>float</td><td>0</td><td>wiener_nonblind_noise</td><td>42.0404183667457</td><td>43.2363776779612</td><td>18.7926389044775</td><td>0.99931942888546</td><td>0.994409035706941</td><td>0.0099582309053251</td></tr><tr><td>eye_blur</td><td>float</td><td>1</td><td>dwdn</td><td>19.6472707095192</td><td>19.9207261301986</td><td>3.35891278055525</td><td>0.637956289836419</td><td>0.63807138068968</td><td>0.148374447358333</td></tr><tr><td>eye_blur</td><td>float</td><td>1</td><td>kerunc</td><td>21.555364704684</td><td>21.968779282732</td><td>4.0215199409868</td><td>0.659070158827585</td><td>0.658371050073044</td><td>0.171202948295039</td></tr><tr><td>eye_blur</td><td>float</td><td>1</td><td>usrnet</td><td>19.8580371724846</td><td>20.2026992831164</td><td>3.84849265377811</td><td>0.628212262997749</td><td>0.625296144798321</td><td>0.170726568756826</td></tr><tr><td>eye_blur</td><td>float</td><td>1</td><td>wiener_blind_noise</td><td>20.8221051262842</td><td>20.7772091561438</td><td>3.00016439667417</td><td>0.652589788596143</td><td>0.652541449667032</td><td>0.125983514766036</td></tr><tr><td>eye_blur</td><td>float</td><td>1</td><td>wiener_nonblind_noise</td><td>20.8221051262842</td><td>20.7772091561438</td><td>3.00016439667417</td><td>0.652589788596143</td><td>0.652541449667032</td><td>0.125983514766036</td></tr><tr><td>eye_blur</td><td>linrgb_16bit</td><td>0</td><td>dwdn</td><td>19.4932004089255</td><td>19.7837914506428</td><td>3.80199828651914</td><td>0.649232667581726</td><td>0.641946323901337</td><td>0.140961730683366</td></tr><tr><td>eye_blur</td><td>linrgb_16bit</td><td>0</td><td>kerunc</td><td>19.7913674256078</td><td>20.3626035376081</td><td>4.84708082126613</td><td>0.570206588656754</td><td>0.573846456646891</td><td>0.216834936584095</td></tr><tr><td>eye_blur</td><td>linrgb_16bit</td><td>0</td><td>usrnet</td><td>18.0247933582547</td><td>18.4764521122878</td><td>4.57829180426167</td><td>0.50895905617295</td><td>0.521663415639418</td><td>0.165591527882829</td></tr><tr><td>eye_blur</td><td>linrgb_16bit</td><td>0</td><td>wiener_blind_noise</td><td>15.2796618298997</td><td>15.3259544310256</td><td>4.32809568805334</td><td>0.47495213936171</td><td>0.47644554043918</td><td>0.189772491723311</td></tr><tr><td>eye_blur</td><td>linrgb_16bit</td><td>0</td><td>wiener_nonblind_noise</td><td>5.51010027002101</td><td>6.40409733881358</td><td>2.60175860399046</td><td>0.0206690790848322</td><td>0.0406650674407494</td><td>0.0573622828077268</td></tr><tr><td>eye_blur</td><td>linrgb_16bit</td><td>1</td><td>dwdn</td><td>19.2096075741877</td><td>19.4615930888965</td><td>3.57688134976279</td><td>0.629323269565413</td><td>0.626628173647006</td><td>0.139571708816115</td></tr><tr><td>eye_blur</td><td>linrgb_16bit</td><td>1</td><td>kerunc</td><td>19.2075873434789</td><td>19.8701616885463</td><td>4.09780487811265</td><td>0.573085514564693</td><td>0.566532365534782</td><td>0.204229264420985</td></tr><tr><td>eye_blur</td><td>linrgb_16bit</td><td>1</td><td>usrnet</td><td>19.4895269078902</td><td>19.9352016561281</td><td>4.06300328088353</td><td>0.66051237155024</td><td>0.644188410405756</td><td>0.160088459378021</td></tr><tr><td>eye_blur</td><td>linrgb_16bit</td><td>1</td><td>wiener_blind_noise</td><td>15.2731535536477</td><td>15.4058646414665</td><td>4.15803130700909</td><td>0.433894893069863</td><td>0.441485855710238</td><td>0.171584598894238</td></tr><tr><td>eye_blur</td><td>linrgb_16bit</td><td>1</td><td>wiener_nonblind_noise</td><td>15.2731535536477</td><td>15.4058646414665</td><td>4.15803130700909</td><td>0.433894893069863</td><td>0.441485855710238</td><td>0.171584598894238</td></tr><tr><td>eye_blur</td><td>srgb_8bit</td><td>0</td><td>dwdn</td><td>19.4563195620283</td><td>19.6662300097505</td><td>3.47450855487508</td><td>0.637715546974601</td><td>0.631863887060043</td><td>0.147006522463194</td></tr><tr><td>eye_blur</td><td>srgb_8bit</td><td>0</td><td>kerunc</td><td>21.6413015035282</td><td>22.2627238492603</td><td>4.45522880728686</td><td>0.661252175067269</td><td>0.666330669387492</td><td>0.169565789597524</td></tr><tr><td>eye_blur</td><td>srgb_8bit</td><td>0</td><td>usrnet</td><td>26.6630431400893</td><td>27.0548160761325</td><td>4.89938924194453</td><td>0.844463901648402</td><td>0.825240645459395</td><td>0.123035248594768</td></tr><tr><td>eye_blur</td><td>srgb_8bit</td><td>0</td><td>wiener_blind_noise</td><td>20.4428450665349</td><td>20.5658975450716</td><td>3.80546065167629</td><td>0.711243659193527</td><td>0.703862504453494</td><td>0.135335478873097</td></tr><tr><td>eye_blur</td><td>srgb_8bit</td><td>0</td><td>wiener_nonblind_noise</td><td>6.14493026166603</td><td>7.26805557010032</td><td>3.28638541563253</td><td>0.0249042149366202</td><td>0.063528731331162</td><td>0.107337962720855</td></tr><tr><td>eye_blur</td><td>srgb_8bit</td><td>1</td><td>dwdn</td><td>19.3750838763536</td><td>19.5688167352068</td><td>3.39988867411746</td><td>0.629396205433827</td><td>0.623734504956573</td><td>0.146506874733982</td></tr><tr><td>eye_blur</td><td>srgb_8bit</td><td>1</td><td>kerunc</td><td>20.8379226143566</td><td>21.339613587438</td><td>3.80978465349817</td><td>0.637654893448805</td><td>0.640059607649014</td><td>0.164139283440934</td></tr><tr><td>eye_blur</td><td>srgb_8bit</td><td>1</td><td>usrnet</td><td>19.2198771123042</td><td>19.6739643601644</td><td>3.81060740247537</td><td>0.607656628236133</td><td>0.604561964730933</td><td>0.16578993995786</td></tr><tr><td>eye_blur</td><td>srgb_8bit</td><td>1</td><td>wiener_blind_noise</td><td>20.5847993916564</td><td>20.5130881057523</td><td>3.33384353910422</td><td>0.665303870384647</td><td>0.665366098741445</td><td>0.128820056434426</td></tr><tr><td>eye_blur</td><td>srgb_8bit</td><td>1</td><td>wiener_nonblind_noise</td><td>20.5847993916564</td><td>20.5130881057523</td><td>3.33384353910422</td><td>0.665303870384647</td><td>0.665366098741445</td><td>0.128820056434426</td></tr><tr><td>gauss_blur</td><td>float</td><td>0</td><td>dwdn</td><td>24.7848164745519</td><td>24.9645458193709</td><td>4.35165842088635</td><td>0.789976177575188</td><td>0.773486211825085</td><td>0.130021596546454</td></tr><tr><td>gauss_blur</td><td>float</td><td>0</td><td>kerunc</td><td>25.4221737720024</td><td>25.7353809655367</td><td>4.28031805719027</td><td>0.791865980851145</td><td>0.778243106490329</td><td>0.134821367372244</td></tr><tr><td>gauss_blur</td><td>float</td><td>0</td><td>usrnet</td><td>28.2866525053603</td><td>28.7441450228077</td><td>5.16210770174133</td><td>0.883631693123102</td><td>0.857824255015552</td><td>0.10351462387507</td></tr><tr><td>gauss_blur</td><td>float</td><td>0</td><td>wiener_blind_noise</td><td>22.3827236826695</td><td>22.6946029595911</td><td>4.12476288640963</td><td>0.79111517356822</td><td>0.778579496207423</td><td>0.120404446144202</td></tr><tr><td>gauss_blur</td><td>float</td><td>0</td><td>wiener_nonblind_noise</td><td>28.4134281803415</td><td>30.4129740803198</td><td>9.13321499681425</td><td>0.950381530390918</td><td>0.937448391643985</td><td>0.0705236218201369</td></tr><tr><td>gauss_blur</td><td>float</td><td>1</td><td>dwdn</td><td>24.6384003093552</td><td>24.7364993285866</td><td>4.19505082913485</td><td>0.7774460823981</td><td>0.763232926255295</td><td>0.130116075003305</td></tr><tr><td>gauss_blur</td><td>float</td><td>1</td><td>kerunc</td><td>25.0379747098647</td><td>25.5062779999242</td><td>4.61134132375731</td><td>0.776934359110944</td><td>0.759945609943993</td><td>0.138678246131143</td></tr><tr><td>gauss_blur</td><td>float</td><td>1</td><td>usrnet</td><td>24.4541906951839</td><td>24.9312523615351</td><td>4.47852368648729</td><td>0.745288060695523</td><td>0.736384276426754</td><td>0.140929460700981</td></tr><tr><td>gauss_blur</td><td>float</td><td>1</td><td>wiener_blind_noise</td><td>22.517170853329</td><td>22.7244467397499</td><td>3.84261209521164</td><td>0.765183996806919</td><td>0.749484567016048</td><td>0.113385259847974</td></tr><tr><td>gauss_blur</td><td>float</td><td>1</td><td>wiener_nonblind_noise</td><td>22.517170853329</td><td>22.7244467397499</td><td>3.84261209521164</td><td>0.765183996806919</td><td>0.749484567016048</td><td>0.113385259847974</td></tr><tr><td>gauss_blur</td><td>linrgb_16bit</td><td>0</td><td>dwdn</td><td>23.5866014665961</td><td>23.8780317987915</td><td>4.02987356743868</td><td>0.804493208499869</td><td>0.784063627004247</td><td>0.123086717575954</td></tr><tr><td>gauss_blur</td><td>linrgb_16bit</td><td>0</td><td>kerunc</td><td>24.4911145606561</td><td>24.440817436571</td><td>4.21008508616606</td><td>0.807421292264137</td><td>0.782133643164216</td><td>0.128749938539352</td></tr><tr><td>gauss_blur</td><td>linrgb_16bit</td><td>0</td><td>usrnet</td><td>21.7869892553906</td><td>21.6068749033598</td><td>4.68619545834288</td><td>0.762598231135632</td><td>0.743977793143814</td><td>0.142327627599629</td></tr><tr><td>gauss_blur</td><td>linrgb_16bit</td><td>0</td><td>wiener_blind_noise</td><td>21.0683755759639</td><td>21.1045924335544</td><td>4.16192565231541</td><td>0.771786656505004</td><td>0.761062338585026</td><td>0.118690306976038</td></tr><tr><td>gauss_blur</td><td>linrgb_16bit</td><td>0</td><td>wiener_nonblind_noise</td><td>4.84815454252162</td><td>5.01361028246542</td><td>0.73155063384927</td><td>0.0115645288880359</td><td>0.012668910286051</td><td>0.0059774305179549</td></tr><tr><td>gauss_blur</td><td>linrgb_16bit</td><td>1</td><td>dwdn</td><td>23.3403029653387</td><td>23.8036476866263</td><td>4.11059586689891</td><td>0.79490551266022</td><td>0.778260510509648</td><td>0.124174984659632</td></tr><tr><td>gauss_blur</td><td>linrgb_16bit</td><td>1</td><td>kerunc</td><td>24.0892884213201</td><td>24.018858294606</td><td>4.0225501763658</td><td>0.787157447594674</td><td>0.76794590150899</td><td>0.12777318658352</td></tr><tr><td>gauss_blur</td><td>linrgb_16bit</td><td>1</td><td>usrnet</td><td>24.1247643651179</td><td>24.233375023656</td><td>4.39119063702996</td><td>0.78060849587885</td><td>0.760470407393975</td><td>0.132202428735228</td></tr><tr><td>gauss_blur</td><td>linrgb_16bit</td><td>1</td><td>wiener_blind_noise</td><td>21.0086320405032</td><td>21.0488551290021</td><td>3.8282309680494</td><td>0.736462431600193</td><td>0.729510170022511</td><td>0.111206267774225</td></tr><tr><td>gauss_blur</td><td>linrgb_16bit</td><td>1</td><td>wiener_nonblind_noise</td><td>21.0086320405032</td><td>21.0488551290021</td><td>3.8282309680494</td><td>0.736462431600193</td><td>0.729510170022511</td><td>0.111206267774225</td></tr><tr><td>gauss_blur</td><td>srgb_8bit</td><td>0</td><td>dwdn</td><td>23.1167284829999</td><td>23.9062758583351</td><td>3.97439890901138</td><td>0.760244384136668</td><td>0.748210079888235</td><td>0.127309919114736</td></tr><tr><td>gauss_blur</td><td>srgb_8bit</td><td>0</td><td>kerunc</td><td>24.0147723495879</td><td>24.7026413530925</td><td>4.23682156803323</td><td>0.775588568830835</td><td>0.753340826429977</td><td>0.132170247731514</td></tr><tr><td>gauss_blur</td><td>srgb_8bit</td><td>0</td><td>usrnet</td><td>26.2742329420996</td><td>26.5480820852158</td><td>4.72817405313665</td><td>0.831176143644644</td><td>0.815129800668294</td><td>0.113953240363504</td></tr><tr><td>gauss_blur</td><td>srgb_8bit</td><td>0</td><td>wiener_blind_noise</td><td>21.737473591111</td><td>22.0753239433263</td><td>3.93954481062157</td><td>0.764185171082335</td><td>0.751790165479783</td><td>0.117188547634191</td></tr><tr><td>gauss_blur</td><td>srgb_8bit</td><td>0</td><td>wiener_nonblind_noise</td><td>5.4470914814406</td><td>5.50566640088377</td><td>0.497950907332461</td><td>0.0148500478807886</td><td>0.0167137906034237</td><td>0.00736963492305571</td></tr><tr><td>gauss_blur</td><td>srgb_8bit</td><td>1</td><td>dwdn</td><td>23.1031718496199</td><td>23.7582295483794</td><td>3.86912223094317</td><td>0.753035768636674</td><td>0.742179380363326</td><td>0.126594582079846</td></tr><tr><td>gauss_blur</td><td>srgb_8bit</td><td>1</td><td>kerunc</td><td>23.6872544780511</td><td>24.2755641876081</td><td>4.13700122974751</td><td>0.751450897923293</td><td>0.735498783315128</td><td>0.132599669863823</td></tr><tr><td>gauss_blur</td><td>srgb_8bit</td><td>1</td><td>usrnet</td><td>23.1173586325069</td><td>23.7474361330788</td><td>4.05334480119239</td><td>0.724758579275199</td><td>0.708927589888982</td><td>0.136021361221164</td></tr><tr><td>gauss_blur</td><td>srgb_8bit</td><td>1</td><td>wiener_blind_noise</td><td>21.9944049647238</td><td>21.9872025058346</td><td>3.71802855659132</td><td>0.749089856634721</td><td>0.735813527245937</td><td>0.110669891311606</td></tr><tr><td>gauss_blur</td><td>srgb_8bit</td><td>1</td><td>wiener_nonblind_noise</td><td>21.9944049647238</td><td>21.9872025058346</td><td>3.71802855659132</td><td>0.749089856634721</td><td>0.735813527245937</td><td>0.110669891311606</td></tr><tr><td>motion_blur</td><td>float</td><td>0</td><td>dwdn</td><td>33.4133474690389</td><td>33.8815714353448</td><td>3.40893476996495</td><td>0.977572302962184</td><td>0.974942914649807</td><td>0.0144788518907227</td></tr><tr><td>motion_blur</td><td>float</td><td>0</td><td>kerunc</td><td>34.0731813422794</td><td>34.2115882918737</td><td>3.81811616105445</td><td>0.976672455513</td><td>0.968023663303728</td><td>0.0231158565609839</td></tr><tr><td>motion_blur</td><td>float</td><td>0</td><td>usrnet</td><td>44.5033274002729</td><td>44.5189852183936</td><td>2.49972418469964</td><td>0.996719657129765</td><td>0.996041919024848</td><td>0.00223587628961164</td></tr><tr><td>motion_blur</td><td>float</td><td>0</td><td>wiener_blind_noise</td><td>26.3264144443209</td><td>26.0053462057116</td><td>4.71933553418085</td><td>0.93094191531229</td><td>0.923900376140682</td><td>0.0403678278886934</td></tr><tr><td>motion_blur</td><td>float</td><td>0</td><td>wiener_nonblind_noise</td><td>45.4477713782716</td><td>62.1080393763319</td><td>39.7914576062949</td><td>0.999807039409637</td><td>0.994692352010589</td><td>0.0104389381290456</td></tr><tr><td>motion_blur</td><td>float</td><td>1</td><td>dwdn</td><td>31.2012333354523</td><td>31.1798568636881</td><td>3.00947935090953</td><td>0.946797573171139</td><td>0.944429973842935</td><td>0.0261350920056733</td></tr><tr><td>motion_blur</td><td>float</td><td>1</td><td>kerunc</td><td>30.7144979633778</td><td>30.7558304736828</td><td>3.39158318671065</td><td>0.926741779006958</td><td>0.920350621486362</td><td>0.0413496357310388</td></tr><tr><td>motion_blur</td><td>float</td><td>1</td><td>usrnet</td><td>25.4038489372979</td><td>25.8727641975945</td><td>4.04035802021759</td><td>0.783894339839983</td><td>0.781908778838828</td><td>0.108969808509974</td></tr><tr><td>motion_blur</td><td>float</td><td>1</td><td>wiener_blind_noise</td><td>26.5315289257875</td><td>25.8857747507974</td><td>3.81877544422752</td><td>0.887025851861</td><td>0.881490930246843</td><td>0.0370703222939418</td></tr><tr><td>motion_blur</td><td>float</td><td>1</td><td>wiener_nonblind_noise</td><td>26.5315289257875</td><td>25.8857747507974</td><td>3.81877544422752</td><td>0.887025851861</td><td>0.881490930246843</td><td>0.0370703222939418</td></tr><tr><td>motion_blur</td><td>linrgb_16bit</td><td>0</td><td>dwdn</td><td>28.2465390214791</td><td>28.6089872700225</td><td>3.87654791884847</td><td>0.940159527285703</td><td>0.932771626130181</td><td>0.0350557360907003</td></tr><tr><td>motion_blur</td><td>linrgb_16bit</td><td>0</td><td>kerunc</td><td>29.0317985504789</td><td>29.4141567734561</td><td>4.08238771282112</td><td>0.941228853902062</td><td>0.929098181846961</td><td>0.047602828655483</td></tr><tr><td>motion_blur</td><td>linrgb_16bit</td><td>0</td><td>usrnet</td><td>26.8894357410442</td><td>26.9693106190396</td><td>4.83213855650212</td><td>0.87539986139746</td><td>0.841449662190436</td><td>0.122981636337352</td></tr><tr><td>motion_blur</td><td>linrgb_16bit</td><td>0</td><td>wiener_blind_noise</td><td>23.1062274037376</td><td>22.7008108594038</td><td>4.41243754075002</td><td>0.843479196015</td><td>0.826244983375723</td><td>0.0971318577477626</td></tr><tr><td>motion_blur</td><td>linrgb_16bit</td><td>0</td><td>wiener_nonblind_noise</td><td>21.9066512077305</td><td>21.721968802894</td><td>3.85409350060438</td><td>0.644556015882641</td><td>0.631782148343549</td><td>0.169224379052034</td></tr><tr><td>motion_blur</td><td>linrgb_16bit</td><td>1</td><td>dwdn</td><td>27.0728568094854</td><td>27.2774368983061</td><td>3.92435749951695</td><td>0.910066378456771</td><td>0.906294034991078</td><td>0.0413766268420869</td></tr><tr><td>motion_blur</td><td>linrgb_16bit</td><td>1</td><td>kerunc</td><td>26.5885420238641</td><td>26.595752716704</td><td>3.63878963340848</td><td>0.867715241711738</td><td>0.860432406307352</td><td>0.0553066193509683</td></tr><tr><td>motion_blur</td><td>linrgb_16bit</td><td>1</td><td>usrnet</td><td>24.3237157540137</td><td>25.0952244992189</td><td>4.16482135124518</td><td>0.809638509822371</td><td>0.794566411237335</td><td>0.101805401317754</td></tr><tr><td>motion_blur</td><td>linrgb_16bit</td><td>1</td><td>wiener_blind_noise</td><td>22.8057401634995</td><td>22.5562424333897</td><td>3.92172764395383</td><td>0.801907163507237</td><td>0.789425207066788</td><td>0.0956796328725214</td></tr><tr><td>motion_blur</td><td>linrgb_16bit</td><td>1</td><td>wiener_nonblind_noise</td><td>22.8057401634995</td><td>22.5562424333897</td><td>3.92172764395383</td><td>0.801907163507237</td><td>0.789425207066788</td><td>0.0956796328725214</td></tr><tr><td>motion_blur</td><td>srgb_8bit</td><td>0</td><td>dwdn</td><td>32.7406617419401</td><td>33.0219711676893</td><td>3.17291565236633</td><td>0.973667567344666</td><td>0.969761753949712</td><td>0.0158990097872699</td></tr><tr><td>motion_blur</td><td>srgb_8bit</td><td>0</td><td>kerunc</td><td>33.8242084059406</td><td>33.7268728794066</td><td>3.53153255034459</td><td>0.971043490713596</td><td>0.963477721688687</td><td>0.0231581071571394</td></tr><tr><td>motion_blur</td><td>srgb_8bit</td><td>0</td><td>usrnet</td><td>40.8209314329745</td><td>40.8786284095191</td><td>2.48690908038635</td><td>0.992181625930786</td><td>0.991190673146158</td><td>0.00448431325031038</td></tr><tr><td>motion_blur</td><td>srgb_8bit</td><td>0</td><td>wiener_blind_noise</td><td>26.633585184133</td><td>26.0331115364226</td><td>4.62353935948714</td><td>0.919823871667862</td><td>0.911639274924596</td><td>0.0467374573227742</td></tr><tr><td>motion_blur</td><td>srgb_8bit</td><td>0</td><td>wiener_nonblind_noise</td><td>29.2590410970772</td><td>28.9999466486703</td><td>4.08924805644867</td><td>0.89297626404953</td><td>0.864995355969481</td><td>0.094423594466291</td></tr><tr><td>motion_blur</td><td>srgb_8bit</td><td>1</td><td>dwdn</td><td>31.3107685210249</td><td>31.2393934600608</td><td>3.42178747246413</td><td>0.959118008595347</td><td>0.951422979883703</td><td>0.0277277127734779</td></tr><tr><td>motion_blur</td><td>srgb_8bit</td><td>1</td><td>kerunc</td><td>29.6585209556111</td><td>29.5148495717892</td><td>3.53969994227123</td><td>0.915409348341009</td><td>0.906836285782233</td><td>0.0481340119002835</td></tr><tr><td>motion_blur</td><td>srgb_8bit</td><td>1</td><td>usrnet</td><td>24.8065002167609</td><td>25.1606981636846</td><td>3.77875154063864</td><td>0.779935257541239</td><td>0.770505094675279</td><td>0.110339999668314</td></tr><tr><td>motion_blur</td><td>srgb_8bit</td><td>1</td><td>wiener_blind_noise</td><td>26.1949468099234</td><td>25.6439188216673</td><td>4.15669227048104</td><td>0.889154568161011</td><td>0.886282896279804</td><td>0.0466807090900329</td></tr><tr><td>motion_blur</td><td>srgb_8bit</td><td>1</td><td>wiener_nonblind_noise</td><td>26.1949468099234</td><td>25.6439188216673</td><td>4.15669227048104</td><td>0.889154568161011</td><td>0.886282896279804</td><td>0.0466807090900329</td></tr></table>