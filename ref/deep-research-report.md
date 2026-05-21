# NABM 중심의 최소 단위지성과 MLP 기반 에이전트 설계 연구 보고서

## 요약

신경 에이전트 기반 모델(Neural Agent-Based Model, NABM)은 전통적 규칙 기반 에이전트(“if-then” 규칙 집합) 대신 **신경망(대표적으로 MLP)을 에이전트 자체로** 두어, 에이전트가 환경·데이터·사회적 상호작용을 통해 학습·적응하도록 만드는 ABM(Agent-Based Model)의 한 계열로 정리할 수 있다. citeturn1view0turn0search16 특히 entity["people","Igor Douven","philosopher of science"]의 NABM 제안은 고전적 의견동역학 ABM인 **Hegselmann–Krause(HK) bounded confidence 모델**의 “사회적 갱신”을, **(i) 파라미터(가중치·편향) 수준의 갱신**과 **(ii) 출력(예측) 수준의 갱신**으로 신경망 에이전트에 이식하는 방식으로 체계화했다. citeturn19view1turn10view0turn3view2

이 관점에서 NABM은 “ABM용 단위지성(unit intelligence)”을 정의하려는 시도—즉, **최소한의 계산 단위가 (1) 내부 표상 업데이트 능력, (2) 사회적 상호작용을 통한 업데이트 능력, (3) 환경에서의 과업 수행 능력**을 동시에 갖도록 만드는 시도—로 해석될 수 있다. Douven의 구성은 이 “최소 단위”를 매우 작고 해석 가능한 MLP로 시작해(얕은 네트워크 포함), 사회적 업데이트 규칙을 명시적으로 설계·실험한다는 점이 특징이다. citeturn11view2turn19view0turn1view0

또한, “Transformer 블록 같은 유의미한 연산 단위”라는 문제의식은 비전·언어 영역에서 **어텐션을 배제한 MLP 기반 ‘믹서/믹싱’ 연산 단위**가 등장·진화한 흐름과 자연스럽게 연결된다. 예컨대 MLP-Mixer는 토큰-믹싱과 채널-믹싱을 분리하고(“공간-관계”와 “특징-변환” 분리) 이를 반복 블록으로 쌓는 방식으로 “블록 단위”를 정의했다. citeturn13view0turn14view0 한편, Transformer의 self-attention은 레이어당 계산복잡도가 \(O(n^2\cdot d)\)로 정리되며, 토큰 길이가 커질수록 병목이 된다. citeturn16view0 이런 맥락에서 “NABM의 사회적 상호작용(에이전트 간 영향)”을 **학습 가능한 믹싱 연산**으로 보고, 이를 **선형·국소·동적**으로 설계하는 연구는 사회동역학 실험·R&D(연구개발/정책 실험)용 시뮬레이터로서의 NABM을 한 단계 확장시키는 핵심 축이 된다. citeturn6search1turn5search1turn6search0

## ABM과 NABM의 정의

ABM(Agent-Based Model)은 **개별 “에이전트”의 행동·상호작용 규칙을 미시적으로 정의**하고, 그 상호작용의 누적 결과로 **거시적 패턴(창발)을 관찰·분석**하는 계산모형으로 요약된다. 응용 분야는 경제·정치·역학·도시계획 등 매우 넓고, 철학·방법론 측면에서도 ABM의 이상화·검증·설명력 문제가 꾸준히 논의되어 왔다. citeturn0search16turn0search27

이 전통적 ABM에서 비판의 한 축은 “에이전트가 너무 단순하다(simplistic)”는 점이다. 규칙 기반 에이전트는 통제·해석이 쉽지만, 실제 인간/조직의 적응·학습·표상 변화를 충분히 담기 어렵다는 지적이 반복되었다. citeturn1view0turn0search27

이 문제의 한 해법으로 Douven은 **ABM과 인공신경망을 결합**해 에이전트가 “인간처럼(더 humanlike) 학습·적응”하는 모델군을 **NABM**으로 제안한다. citeturn1view0 흥미로운 점은, Douven이 NABM을 특정 신경망/특정 ABM에 고정하지 않고 “매우 넓은 제안”으로 두면서도, 최초의 구체적 실험틀로 **HK 모델(의견동역학의 대표적 bounded confidence ABM)**을 선택했다는 것이다. citeturn1view0turn9search4

HK 모델은 (요약하면) 각 에이전트가 **자신과 충분히 가까운(신뢰/유사성 임계값 \(\varepsilon\) 내) 이웃의 의견을 평균**하는 방식으로 의견을 갱신하고, \(\varepsilon\)에 따라 합의·분극·파편화 등이 나타날 수 있는 것으로 널리 연구되어 왔다. citeturn9search4turn9search8 Douven의 핵심은 이 “bounded confidence + 평균 갱신”을 **신경망의 ‘파라미터 평균’ 또는 ‘출력 평균’**으로 번역해, 규칙 기반 갱신이 아니라 **학습하는 에이전트의 사회적 학습**을 실험할 수 있게 만든 점이다. citeturn19view1turn10view0turn3view2

## Douven의 NABM 설계와 실험

Douven의 NABM 논문은 사회적 학습을 “개별 학습(환경 데이터에 대한 학습)”과 “사회적 갱신(동료로부터의 영향)”의 결합으로 보고, HK 모델의 두 핵심 요소—(a) 동료 선택(peerhood, \(\varepsilon\)에 의해 규정), (b) 사회 요인과 환경 요인의 혼합(가중치 \(\alpha\))—을 신경망 에이전트로 이식한다. citeturn3view1turn1view0

Douven이 특히 강조하는 연결고리는, MLP 에이전트에서 “상태(state)”와 “출력(output)”이 분리된다는 점이다. 동일 시점의 에이전트는 (1) 내부 상태로서 **가중치·편향(파라미터)**을 가지며, (2) 과업이 주어졌을 때의 **예측/판단(출력)** 또한 가진다. 이 구분 덕분에 **(i) 상태 유사성 기반 동료 선택**과 **(ii) 출력 유사성 기반 동료 선택**이 모두 가능해지고, 이 둘을 결합한 모델도 만들 수 있다. citeturn3view1turn19view1

상태 기반(peerhood) 유사성은 파라미터 벡터를 만든 뒤 **코사인 유사도**로 정의된다. Douven은 에이전트 \(i\)의 가중치·편향을 벡터 \(\mathrm{params}_i\)로 모으고,
\[
\mathrm{cossim}(i,j)=\frac{\mathrm{params}_i\cdot \mathrm{params}_j}{\lVert \mathrm{params}_i\rVert\ \lVert \mathrm{params}_j\rVert}
\]
로 유사도를 계산한다. citeturn19view0 그리고 **state-based peers**는 \(\mathrm{cossim}(i,j)\ge 1-\varepsilon\)일 때로 정의한다(즉 \(\varepsilon\)가 커질수록 동료 조건이 느슨해짐). citeturn19view1

이 위에서 Douven은 세 가지 NABM 갱신 방식을 제시한다.

첫째, **상태 기반(state-based) 사회 갱신**은 (개념적으로) “파라미터 평균 → 개별 학습 → 혼합 평균”의 순서로 요약된다. 구체적으로는 (a) 커뮤니티 내 모든 에이전트의 코사인 유사도 행렬을 계산해 각 에이전트의 동료를 선택하고, (b) 동료들의 파라미터 평균을 저장한 뒤, (c) 각 에이전트를 데이터로 한 라운드 학습(환경/세계 요인)시키고, (d) 학습 후 자기 파라미터와 동료 평균 파라미터를 \(\alpha\)로 가중 평균해 새 파라미터로 설정한다. citeturn3view2turn18view1

둘째, **출력 기반(output-based) 사회 갱신**은 “개별 학습 → 예측 생성 → 예측 유사성으로 동료 선택 → 예측의 (가중)평균” 구조를 가진다. Douven은 출력이 반드시 수치 스칼라일 필요는 없고(분류 확률분포 등), 따라서 ‘평균’ 연산은 과업/출력의 성격에 맞게 정의되어야 한다고 명시한다. citeturn10view0turn11view2

셋째, **결합(combined) 갱신**은 상태 기반(파라미터 평균)과 출력 기반(예측 평균)을 모두 수행한다. 이 경우 동료선정·혼합을 위한 파라미터가 \((\varepsilon_1,\alpha_1)\) (상태 기반)과 \((\varepsilon_2,\alpha_2)\) (출력 기반)로 분리된다. Douven은 결합 모델을 단계적으로 “(1) 상태기반 동료 파라미터 평균, (2) 학습, (3) 파라미터 혼합, (4) 예측 생성, (5) 출력기반 동료 선택, (6) 예측 혼합”으로 제시한다. citeturn10view0turn3view2

이 설계를 검증하기 위해 Douven은 두 가지 계산 실험을 수행한다.

첫 번째 실험은 **색채 분류(개념 형성/범주 학습)** 과업이다. 에이전트(MLP)는 색상 유사공간(특히 CIELUV)의 좌표를 입력으로 받아 색 범주를 분류하도록 학습되며, 훈련·평가 데이터는 World Color Survey(WCS)의 320개 Munsell chip에서 구성된다. Douven은 실제 색 이름짓기 연구에서 기본색 용어 사용과 개인차가 존재한다는 점을 언급하고, 목표 범주(“자연스러운 색 개념”에 대한 근사)를 만들기 위해 k-means를 사용한다. citeturn4view1turn10view1

두 번째 실험은 **고혈압 단계 예측(의학적 진단 과업)**이다. 에이전트는 환자 특성(나이, 성별, BMI, 당뇨 상태, 신체활동, 음주, 흡연 등)을 바탕으로 고혈압 단계에 대한 확률분포를 예측하며, 훈련 데이터는 NHANES(2017–2020년 3월까지 수집분)에서 구성된다. citeturn11view1turn4view2 이 실험에서 Douven은 확률분포 출력의 유사성 측도로 Jensen–Shannon divergence를 사용하고, 출력 평균을 “함수의 평균(선형 풀링)”으로 정의한다. citeturn11view2

결과적으로 Douven은 두 과업 모두에서 **사회 갱신(특히 결합 갱신)**이 비사회적(개별) 업데이트 대비 학습 성능을 유의미하게 개선함을 보고한다. 예를 들어 색채 분류 실험에서는 에폭별 mutual information 곡선을 제시하며 사회 업데이트 방식들이 더 높은 성능을 보인다고 논의한다. citeturn11view0 고혈압 과업에서는 AULC(learning curve 아래 면적) 비교에서 결합 갱신이 다른 방식보다 높고, 결합 vs 출력기반 사이에서도 큰 효과크기를 보고한다. citeturn11view1

## MLP 기반 ‘연산 단위’와 NABM의 연결

NABM을 “ABM용 단위지성”으로 보려는 질문은, 딥러닝 아키텍처 설계에서 “유의미한 블록(연산 단위)”를 무엇으로 둘 것인가라는 질문과 구조적으로 닮아 있다. Transformer는 self-attention과 position-wise FFN을 핵심 블록으로 반복하며, self-attention의 레이어 복잡도는 \(O(n^2\cdot d)\)로 제시된다. citeturn16view0 반대로, 비전 영역에서는 self-attention 없이도 경쟁력 있는 성능을 내기 위한 다양한 “MLP-블록”이 제안되면서, 블록 수준에서의 **“믹싱(mixing)”**이 하나의 설계 축으로 부상했다. citeturn13view0turn5search1

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["MLP-Mixer architecture diagram token mixing channel mixing","CycleMLP Cycle FC layer diagram","Vision Permutator Permute-MLP block diagram","Hegselmann-Krause bounded confidence model diagram"],"num_per_query":1}

MLP-Mixer는 입력을 “패치(토큰) × 채널” 테이블 \(X\in\mathbb{R}^{S\times C}\)로 보고, **token-mixing MLP(공간/토큰 간 정보 혼합)**와 **channel-mixing MLP(채널/특징 간 변환)**를 교차로 적용하는 블록을 반복한다. citeturn13view0turn14view0 논문은 token-mixing과 channel-mixing의 hidden width를 조절해 계산 복잡도를 “패치 수에 대해 선형”으로 설계할 수 있다고 설명하며, 전형적 ViT의(제한 없는) self-attention이 가지는 제곱 복잡도와 대비시킨다. citeturn14view1turn16view0 다만 MLP-Mixer는 해상도(패치 수 \(S\))가 바뀌면 token-mixing 가중치 행렬 차원이 바뀌는 구조적 문제가 있어, 해상도 변경 시 블록을 조정해야 함을 명시한다(논문은 block-diagonal 초기화 등 특정 조정법을 제시). citeturn14view2

이 “입력 크기/해상도 결합” 문제는 후속 Vision-MLP 연구의 주요 동기가 되었고, CycleMLP는 기존 MLP류(MLP-Mixer, ResMLP, gMLP 등)가 이미지 크기와 강하게 결합되어 객체탐지·세그멘테이션 같은 dense prediction에 직접 쓰기 어렵다는 점을 지적하며, **가변 입력 스케일을 다루는 Cycle FC 기반 구조**를 제안한다. citeturn5search1turn5search5 Vision Permutator(ViP)는 2D 위치정보를 보존하기 위해 **height/width 축을 분리해 선형 투영**을 수행하고(permute-MLP), 장거리 의존성 포착을 어텐션 없이 달성하려는 흐름으로 이해할 수 있다. citeturn5search10turn5search6 AS-MLP는 채널을 축 방향으로 shift해 국소 의존성을 흡수하고, detection/segmentation 등 하류 과업 적용을 강조한다. citeturn5search3turn5search7 Hire-MLP는 **지역(Region) 내부 재배치와 지역 간 순환 이동(circular shift)**을 통해 국소-전역 문맥을 모두 다루는 “계층적 재배치”를 제안한다. citeturn6search2turn6search6

정적(입력과 무관한) 공간 믹싱의 한계—“내용과 상관없이 동일 가중치로 토큰을 섞는다”—를 직접 겨냥한 방법도 등장했다. DynaMixer는 토큰 내용을 활용해 **동적으로 mixing matrix를 생성**한다고 주장한다. citeturn6search1 Wave-MLP는 토큰을 “진폭+위상”을 갖는 파동으로 보고, phase-aware token mixing(PATM)을 통해 토큰 관계를 입력 의미에 따라 동적으로 조절하는 아이디어를 제시한다. citeturn6search0turn6search24 HyperMixer는 NLP 문맥에서 “정적 토큰 믹싱이 유도편향을 충분히 담지 못한다”는 문제의식 아래, hypernetwork로 토큰 믹싱 MLP 자체를 동적으로 생성하는 변형을 제안한다. citeturn0academia35 gMLP는 gating이 포함된 MLP 구조로, 비전·언어에서 self-attention 없이도 경쟁력 있는 성능이 가능하다는 방향을 제시한다. citeturn6search3turn6search7

이 흐름을 NABM으로 번역하면, “토큰-믹싱”은 ABM에서의 **사회적 상호작용(이웃/동료로부터의 영향)을 계산하는 연산**, “채널-믹싱”은 **에이전트 내부의 표상 변환/추론(개별 학습)을 수행하는 연산**으로 대응시킬 수 있다(해석). Douven의 NABM이 “파라미터 평균(상태 기반)”과 “예측 평균(출력 기반)”을 명시적으로 구분한 것은, 바로 이 “사회적 믹싱 연산”을 무엇으로 정의할지(내부 상태를 섞을지, 행동/판단을 섞을지)를 ABM 설계 변수로 끌어올린 사례로 볼 수 있다. citeturn19view1turn10view0turn3view2

## NABM로 수행하는 사회동역학·R&D 실험 설계

사용자가 말한 “RD”는 문맥상 R&D(연구개발/정책 실험)로도, 혹은 사회동역학의 특정 수리동역학(replicator dynamics 등)로도 해석될 수 있다. 본 절에서는 **(a) R&D/정책 실험을 위한 시뮬레이션**, **(b) 사회동역학(의견·행동 확산) 실험**이라는 두 축을 중심으로 NABM의 실험 설계 패턴을 정리한다.

첫째, NABM의 강점은 “사회적 상호작용(네트워크/동료 선택)”과 “개별 학습(데이터/환경)”이 동시에 작동할 때의 결과를, **모형 내부 메커니즘까지 추적하며** 실험할 수 있다는 점이다. Douven은 동일한 HK 기반 틀에서 state-based / output-based / combined 업데이트를 직접 비교하고, (색채 분류, 고혈압 예측)처럼 서로 다른 형태의 과업에서 사회 갱신이 성능에 미치는 효과를 계량한다. citeturn11view1turn11view0turn10view0 즉 NABM은 단순히 “의견이 평균으로 수렴한다” 수준을 넘어, **학습되는 분류기/진단기 자체가 사회적으로 동조·다양화되는 과정**을 시뮬레이션 객체로 만든다. citeturn11view2turn19view1

둘째, 사회동역학 실험 측면에서는 HK/Deffuant류 bounded confidence 모델이 “합의·양극화·클러스터링”을 연구하는 표준 도구로 자리 잡아 왔고, \(\varepsilon\) 임계값과 네트워크 구조가 거시 결과를 바꾸는 것이 반복적으로 분석되어 왔다. citeturn9search4turn9search8turn9search9 NABM은 여기에 “에이전트 내부가 학습기”라는 층을 추가한다. 예컨대 다음과 같은 실험 질문이 자연스럽다: 동료선정이 “상태 유사성(파라미터)”일 때와 “출력 유사성(행동/판단)”일 때, 분극/파편화의 임계 조건이 어떻게 이동하는가? Douven은 상태 기반 동료를 \(\mathrm{cossim}(i,j)\ge 1-\varepsilon\)로 정의하며, \(\varepsilon\)의 변화가 동료 포함 범위를 바꾼다는 HK적 직관을 유지한다. citeturn19view1turn3view1 출력 기반 동료는 과업에 맞춘 유사도(예: 분류에서는 분류 유사도, 확률분포 출력에서는 JS divergence)로 정의될 수 있고, 이는 “사회적 상호작용의 의미”를 과업 수준에서 바꾼다. citeturn11view2turn10view0

셋째, R&D/정책 실험에서는 ABM이 **개입(intervention) 시나리오를 가상으로 평가**하는 도구로 실무적으로도 활용되어 왔다. COVID-19 맥락에서 “정책 믹스 평가”, “접촉추적/격리” 등을 시뮬레이션하기 위한 detailed ABM이 공개·운영된 사례(OpenABM-Covid19 등)가 있고, 정책결정에 필요한 불확실성·계산효율·사용성을 모두 요구한다는 점이 강조된다. citeturn17search12turn17search4 또한 전 세계의 COVID-19 대응에서 시뮬레이션이 중요한 역할을 했다는 르포성 정리도 존재한다. citeturn17search6turn17search2 이 축에서 NABM은 “정책 개입이 ‘사람의 학습/습관화/순응’에 의해 시간이 지나며 변한다”는 요소를 포함시키는 방향으로 확장될 수 있다(제안). Douven이 제시한 고혈압 예측 실험은, “현실 데이터 기반 개별학습 + 사회적 갱신”이라는 형태가 정책·의료·교육 같은 R&D 과업으로 이식될 수 있음을 보여주는 최소 예시로 읽힌다. citeturn11view1turn11view2

넷째, NABM을 실제 데이터에 맞추고(캘리브레이션) 실험 신뢰도를 올리는 것은 별도의 핵심 과제다. ABM은 대체로 우도함수가 난해해 시뮬레이션 기반 추론(SBI)을 쓰는 경우가 많고, 미시상태(에이전트 상태·상호작용 로그) 자체를 활용하는 방법이 중요한 연구 주제가 되었다. Dyer 등은 ABM의 미시 데이터가 그래프/시계열 구조를 갖는다는 점을 활용해 temporal graph neural network로 ABM 파라미터의 posterior를 직접 학습하는 접근을 제안한다. citeturn7search0turn7search4 더 넓게는 “ML-assisted ABM”을 정리한 설문들도, 대리모형(surrogate), 캘리브레이션, 정책 탐색, 시뮬레이션 가속 등 다양한 결합 양식을 체계화한다. citeturn7search1

마지막으로, 최근 2–3년 사이에는 대규모 언어모델(LLM)을 “에이전트 두뇌”로 쓰는 생성형 사회 시뮬레이션이 급속히 확산되었다(Generative Agents 등). citeturn7search3turn7search30 이는 NABM(MLP 에이전트)보다 훨씬 풍부한 행동 레퍼토리를 제공하지만, 계산비용·검증·편향 문제도 동시에 크게 만든다(설문에서 도전과제 논의). citeturn7search30 또한 LLM 다중에이전트를 DAG로 조직해 1,000+ 에이전트를 스케일링하며 “협업 스케일링 법칙”을 보고한 MacNet 연구는, 상호작용 토폴로지·컨텍스트 폭발 억제 같은 시스템 설계 변수가 집단지성에 직접 영향을 준다는 점을 보여준다. citeturn8search2turn8search3turn8search6 이 흐름은 “NABM을 어떻게 대규모로 조직할 것인가”라는 질문에, 비록 다른 기술 스택이지만 유용한 시사점을 제공한다.

## 차세대 ‘단위지성’로서 NABM을 고도화하는 로드맵

NABM을 Transformer 블록처럼 “유의미한 최소 단위지성”으로 정식화하려면, 단위를 (A) 에이전트 내부 연산, (B) 사회적 믹싱 연산, (C) 학습·갱신 규칙의 조합으로 재정의하는 것이 핵심이다. Douven의 NABM은 (B)를 파라미터 평균/예측 평균로 명시화했고, (C)를 \(\varepsilon,\alpha\) 같은 제어 파라미터로 실험 가능하게 만들었다는 점에서 이미 “단위 정의”의 성격을 갖는다. citeturn19view1turn10view0turn3view2

다만 이 단위가 “Transformer급 의미”를 갖기 위해 남는 연구 의제는 분명하다.

첫째, **사회적 믹싱의 동적화**다. Douven의 state-based는 동료 집합을 코사인 유사도로 고르고 평균을 취하지만, “누구의 영향을 얼마나 받을지”를 입력 상황에 따라 동적으로 조절하는 메커니즘은 제한적이다. citeturn19view0turn3view2 비전 MLP가 정적 토큰 믹싱의 한계를 넘어 DynaMixer(동적 mixing matrix)나 Wave-MLP(위상 기반 동적 모듈레이션)로 발전한 것처럼, NABM에서도 “동료 가중치”를 학습 가능한 함수로 올리거나, 상황/신뢰/목표에 따라 가변화하는 설계가 자연스럽다. citeturn6search1turn6search0

둘째, **단위 내부의 최소 구조를 무엇으로 둘지**다. Douven은 매우 얕은 MLP로도 논의가 일반화될 수 있다고 말하지만, 실제 사회적 의사결정 과업(기억·지연 효과·규범 학습 등)을 담으려면 gating, 메모리, 혹은 더 강한 표현(예: gMLP의 gating, HyperMixer의 동적 믹서 등)이 필요할 수 있다. citeturn4view1turn6search3turn0academia35 여기서 “최소 단위”의 정의는 단지 네트워크 구조가 아니라, **어떤 상태변수를 보유하고 어떻게 업데이트하는지**(예: “belief state”, “trust state”, “goal state”)까지 포함해야 한다(제안). Douven이 state(파라미터)와 output(예측)을 구분한 것은 이 방향의 첫걸음으로 볼 수 있다. citeturn3view1turn19view1

셋째, **사회적 학습의 두 수준(파라미터/출력) 사이의 정합성** 문제다. Douven의 “상태 기반 갱신”은 파라미터를 평균해 동료의 ‘내부 상태’를 섞는 방식이며, 연합학습에서의 iterative model averaging(FedAvg)과 구조적 유사성이 있다(해석). citeturn3view2turn12search0 반대로 “출력 기반 갱신”은 예측값을 평균/절충하는 방식으로, 앙상블 예측을 평균하거나 이를 하나의 모델로 증류하는 지식증류(knowledge distillation) 관점과 연결될 수 있다(해석). citeturn10view0turn12search1 NABM 연구는 이 둘이 언제 상충하고 언제 보완하는지, 그리고 결합(combined)에서의 안정적 학습 조건이 무엇인지(예: \(\alpha_1,\alpha_2\) 설정, 동료선정 노이즈, 데이터 분산)를 더 체계적으로 밝혀야 한다. citeturn10view0turn11view1

넷째, **검증·설명·캘리브레이션**의 방법론이다. ABM은 이상화와 적합성 문제로 비판을 받아 왔고, NABM은 여기에 “신경망의 불투명성”까지 추가된다. citeturn0search27turn1view0 따라서 NABM이 R&D/사회동역학 실험의 도구로 유통되려면, (a) 민감도 분석과 파라미터 탐색, (b) 데이터 기반 캘리브레이션, (c) 관측 가능한 지표(거시 통계, 미시 로그)의 사전 합의가 필요하다. Dyer 등이 제안한 GNN 기반 SBI는 “미시 상태를 직접 활용해 캘리브레이션한다”는 방향에서 특히 NABM에도 바로 결합 가능한 도구상자다. citeturn7search0turn7search4

종합하면, Douven의 NABM은 “MLP 에이전트 + 사회적 믹싱(상태/출력) + HK식 동료선정(\(\varepsilon\))과 혼합(\(\alpha\))”을 통해 **ABM 단위지성의 최소 정의를 실험 가능한 형태로 제시**한 사례다. citeturn19view1turn10view0turn11view1 차세대 연구의 핵심은 이 최소 단위를 (1) 더 동적으로, (2) 더 다양한 과업·환경으로, (3) 더 엄밀한 캘리브레이션과 검증 위로 확장하면서도, “단위지성”이라는 개념적 단순성을 잃지 않는 설계를 찾는 데 있다. citeturn7search1turn6search1turn6search0