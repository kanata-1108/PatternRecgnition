\section{ロジスティック回帰}
ロジスティック回帰とは、二値分類に用いられるアルゴリズムであり、あるデータにおいてそのデータが2つのクラスのどちらに所属するか、その確率を予測する。

以下に二次元におけるクラス0に所属する確率$P_{0}$、クラス1に所属する確率$P_{1}$を示す。またここで示す$t_{n}(1\leqq n \leqq N)$とはデータのラベルであり、0または1を示す。
\begin{align}
  \nonumber
  \begin{cases}
    t_{n} = 0 & P(x_{n}, y_{n})\\
    t_{n} = 1 & 1 - P(x_{n}, y_{n})
  \end{cases}
\end{align}

上記の確率は以下のように一つの式で表すことができる。説明において$P(x_{n}, y_{n})$を$z_{n}$とおいて書き換えた式も示す。

\begin{align}
  P_{n} &= P(x_{n}, y_{n})^{t_{n}}\{1 - P(x_{n}, y_{n})\}^{1 - t_{n}}\nonumber\\
        &= z_{n}^{t_{n}}{(1 - z_{n})}^{1 - t_{n}}
\end{align}

また、前述したように出力される値は確率であるため、0$\sim$1に変換する必要があり、それを行うのが以下に示すロジスティック関数と呼ばれる関数である。
\begin{align}
  \sigma(a) = \frac{1}{1 + e^{-a}}
\end{align}

ロジスティック回帰では、ロジスティック関数の引数$a$に重み$\bm w(w_{0}, w_{1},\cdots, w_{M})$とデータ$\bm \phi(\phi_{1}, \phi_{2},\cdots, \phi_{N})$の内積（重み付け和）を取る。

式(2)を書き換えると以下のような式で表すことができる。
\begin{align}
  \sigma(\bm w^{T} \phi_{n}) = \frac{1}{1 + e^{-(\bm w^{T} \phi_{n})}}
\end{align}

ロジスティック関数より、入力されたデータから確率を出力することが可能になったため
\begin{align}
  P(x_{n}, y_{n}) = z_{n} = \sigma(\bm w^{T} \phi_{n}) = \frac{1}{1 + e^{-(\bm w^{T} \phi_{n})}}
\end{align}
のように表すことができる。

ここで、データ\{$\bm \phi$, t\}であり、これに対する尤度関数は以下の式で表される。
\begin{align}
  P &= \prod_{n = 1}^{N}P_{n}\nonumber\\
    &= \prod_{n = 1}^{N}z_{n}^{t_{n}}{(1 - z_{n})}^{1 - t_{n}}\nonumber\\
    &= \prod_{n = 1}^{N}\sigma(\bm w^{T} \phi_{n})^{t_{n}}{(1 - \sigma(\bm w^{T} \phi_{n}))}^{1 - t_{n}}\nonumber
\end{align}

この尤度$P$を最大化するようなパラメータ$\bm w$を求めることがロジスティック回帰の目的である。

ここで、誤差関数$E(\bm w)$を以下のように定義することで、誤差$E$を最小化するパラメータ$\bm w$を求める問題と言い換えられる。
\begin{align}
  E(\bm w) &= \log P\nonumber\\
           &= -\log \prod_{n = 1}^{N}z_{n}^{t_{n}}{(1 - z_{n})}^{1 - t_{n}}\nonumber\\
           &= -\log \prod_{n = 1}^{N}\sigma(\bm w^{T} \phi_{n})^{t_{n}}{(1 - \sigma(\bm w^{T} \phi_{n}))}^{1 - t_{n}}\nonumber
\end{align}
\newpage
\section{パラメータ更新手順}
1章の式(7)で定義した誤差関数を以下のように展開する
\begin{align}
  E(\bm w) &= -\log \prod_{n = 1}^{N}z_{n}^{t_{n}}{(1 - z_{n})}^{1 - t_{n}}\nonumber\\
           &= -\log \prod_{n = 1}^{N}z_{n}^{t_{n}} - \log \prod_{n = 1}^{N}{(1 - z_{n})}^{1 - t_{n}}\nonumber\\
           &= -t_{n}\log \prod_{n = 1}^{N}z_{n} - (1 - t_{n})\log \prod_{n = 1}^{N}(1 - z_{n})\nonumber\\
           &= - \sum_{n = 1}^{N}t_{n}\log z_{n} - \sum_{n = 1}^{N}(1 - t_{n})\log (1 - z_{n})\nonumber\\
           &= - \sum_{n = 1}^{N} \{t_{n}\log z_{n} + (1 - t_{n})\log (1 - z_{n})\}\nonumber
\end{align}
上記の誤差$E$を最小化したいので、$\displaystyle\frac{\partial E(\bm w)}{\partial \bm w} = 0$となるパラメータ$\bm w$をニュートン法によって求めることを考える。

ニュートン法では以下の式によってパラメータ$w$を求める。
\begin{align}
  \bm w^{(new)} = \bm w^{(old)} - \bm H^{-1}\nabla E(\bm w)
\end{align}

ここで$w^{(new)}$は更新後のパラメータを示し、$w^{(old)}$は更新前のパラメータを示す。また、$\bm H$はヘッセ行列を示し二階微分を要素とする行列である。ヘッセ行列は以下のように書くことができる
\begin{align}
  \bm H = \nabla\nabla E(\bm w) = \frac{\partial^{2} E(\bm w)}{\partial \bm w \partial \bm w^{T}}\nonumber
\end{align}
パラメータを更新するにあたって$\nabla E(\bm w)$と$\bm H$を求める必要があるため、以下に$\nabla E(\bm w)$と$\bm H$を求める過程を示す
\begin{align}
  \nabla E(\bm w) &= \frac{\partial E(\bm w)}{\partial \bm w}\nonumber\\
                  &= -\sum_{n = 1}^{N}\frac{\partial E(\bm w)}{\partial z_{n}}\frac{\partial z_{n}}{\partial \bm w}\\
                  &= -\sum_{n = 1}^{N}(\frac{t_{n}}{z_{n}} - \frac{1 - t_{n}}{1 - z_{n}})z_{n}(1 - z){n}\phi_{n}\nonumber\\
                  &= -\sum_{n = 1}^{N}(t_{n} - z_{n})\phi_{n}\nonumber\\
                  &= \sum_{n = 1}^{N}(z_{n} - t_{n})\phi_{n}\nonumber\\
                  &= \Phi^{T}(\bm z - \bm t)
\end{align}
\newpage
式(6)の$\displaystyle\frac{\partial z_{n}}{\partial \bm w}$についてこれを解く方法を以下に示す。
$z_{n}$は式(4)でも示した通り引数に重み$\bm w$とデータ$\bm \phi$の内積（重み付け和）を取るロジスティック関数である。
\begin{align}
  \frac{\partial z_{n}}{\partial \bm w} &= \frac{\partial \displaystyle\frac{1}{1 + e^{(\bm w^{T}\phi_{n})}}}{\partial \bm w}\nonumber\\
                                        &= \frac{\partial \displaystyle\frac{1}{1 + e^{(\bm w^{T}\phi_{n})}}}{\partial \bm w^{T}\phi_{n}}\frac{\partial \bm w^{T}\phi_{n}}{\partial \bm w}\nonumber\\
                                        &= \frac{1}{1 + e^{(\bm w^{T}\phi_{n})}}\Bigl(1 - \frac{1}{1 + e^{(\bm w^{T}\phi_{n})}}\Bigr)\phi_{n}\nonumber\\
                                        &= z_{n}(1 - z_{n})\phi_{n}\nonumber
\end{align}

次にヘッセ行列$H$を求める過程を示す。
\begin{align}
  \bm H &= \nabla\nabla E(\bm w)\nonumber\\
        &= \frac{\partial^{2} E(\bm w)}{\partial \bm w \partial \bm w^{T}}\nonumber\\
        &= \frac{\partial}{\partial \bm w^{T}}\frac{\partial E_(\bm w)}{\partial \bm w}\nonumber
        \intertext{ここで、$\displaystyle\frac{\partial E(\bm w)}{\partial \bm w}$は$\displaystyle\sum_{n = 1}^{N}(z_{n} - t_{n})\phi_{n}$であったため}
        &= \sum_{n = 1}^{N}\frac{\partial}{\partial \bm w^{T}}(z_{n} - t_{n})\phi_{n}\nonumber\\
        &= \sum_{n = 1}^{N}\frac{\partial (z_{n} - t_{n})\phi_{n}}{\partial \bm w^{T}}\nonumber\\
        &= \sum_{n = 1}^{N}\frac{\partial z_{n}\phi_{n}}{\partial \bm w^{T}}\nonumber\\
        &= \sum_{n = 1}^{N}z_{n}(1 - z_{n})\phi_{n}\phi^{T}_{n}
\end{align}
\begin{align}
  \intertext{式(8)は以下のように3つの行列に分解し、表すことができる。}
  &= \begin{pmatrix}
      \phi_{1,0} & \phi_{2,0} & \cdots & \phi_{N,0}\\
      \phi_{1,1} & \phi_{2,1} & \cdots & \phi_{N,1}\\
      \vdots & \vdots & \ddots & \vdots\\
      \phi_{1,M} & \phi_{2,M} & \cdots & \phi_{N,M}
     \end{pmatrix}
          \begin{pmatrix}
      z_{1}(1 - z_{1}) & 0 & \cdots & 0\\
      0 & z_{2}(1 - z_{2}) & \cdots & 0\\
      \vdots & \vdots & \ddots & \vdots\\
      0 & 0 & \cdots & z_{N}(1 - z_{N})
     \end{pmatrix}
     \begin{pmatrix}
      \phi_{1,0} & \phi_{1,1} & \cdots & \phi_{1,M}\\
      \phi_{2,0} & \phi_{2,1} & \cdots & \phi_{2,M}\\
      \vdots & \vdots & \ddots & \vdots\\
      \phi_{N,0} & \phi_{N,1} & \cdots & \phi_{N,M}\nonumber
     \end{pmatrix}
  \intertext{中央の対角行列を$R$とすると}
  &= \Phi^{T} R \Phi
\end{align}
\begin{align}
  \intertext{式(7),(9)を式(5)に代入すると以下のように表すことができる。}
  &= \bm w^{(new)} = \bm w^{(old)} - (\Phi^{T} R \Phi)^{-1}\Phi^{T}(\bm z - \bm t)
  \intertext{このようにしてロジスティック回帰におけるパラメータの更新を行う}\nonumber
\end{align}