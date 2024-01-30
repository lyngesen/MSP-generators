### Latex config files

* `commands.tex` - File containing abbriviations, notation and commands used in latex project.
* `generate_readme.py` - File for generating markdown files `README.md` from `commands.tex`.
* `literature.bib` - Bibtex file containing references.
* `logo_AU.png` - Logo for Aarhus University.
* `preamble.tex` - preamble for latex project.
* `README.md` - This readme file.


<!--START NOTATION-->
# Abbriviations
* *MO* -  Multi-objective
* *MOCO* -  Multi-objective combinatorial optimization
* *BOCO* -  Bi-objective combinatorial optimization
* *MO-MSP* -  Multi-objective Minkowski sum optimization problem
# Commands
## Ehrgott set notation
| Symbol |Shortcut|Command|Description| 
|-----|:------|:--------|:----------| 
|$\mathcal{X}$| `\X` | `\mathcal{X}` |  Decision space|
|$\mathcal{X}_E$| `\Xe` | `\mathcal{X}_E` |  Effecient solutions|
|$\mathcal{Y}$| `\Y` | `\mathcal{Y}` |  Objective space points|
|$\mathcal{Y}_{\mathcal{N}}$| `\Yn` | `\mathcal{Y}_{\mathcal{N}}` |  Non-dominated points|
|$\mathcal{L}$| `\L` | `\mathcal{L}` |  Lower bound set|
|$\mathcal{U}$| `\U` | `\mathcal{U}` |  Upper bound set|
## Article notation 
| Symbol |Shortcut|Command|Description| 
|-----|:------|:--------|:----------| 
|$\mathcal{N}$| `\mcN` | `\mathcal{N}` |  Non-dominated set|
|$\bar{\mathcal{Y}}^s$| `\YsS` | `\bar{\mathcal{Y}}^s` |  Conditioned non-dominated set|
|$\bar{\mathcal{Y}}^{1}$| `\Yc` | `\bar{\mathcal{Y}}^{1}` |  a minimal generator|
|$\bar{\mathcal{Y}}^{s}$| `\Ycs` | `\bar{\mathcal{Y}}^{s}` |  a minimal generator wrt. several subproblems.|
|$\hat{\mathcal{Y}}^{1}$| `\hY` | `\hat{\mathcal{Y}}^{1}` |  subset of non-dominated points|
|$\ ^g\bar{\mathcal{Y}}^{1}$| `\Yc` | `\ ^g\bar{\mathcal{Y}}^{1}` |  a minimal generator|
|$\hat{\mathcal{Y}}^{1}$| `\hYc` | `\hat{\mathcal{Y}}^{1}` |  (another) minimal generator|
|$\bar{\mathcal{U}}^{1}$| `\Uc` | `\bar{\mathcal{U}}^{1}` |  a conditional upper bound set|
|$\bar{\mathcal{Y}}$| `\bY` | `\bar{\mathcal{Y}}` |  bar over objective space|
|$\bar{\mathcal{Y}}_\mathcal{N}$| `\bYn` | `\bar{\mathcal{Y}}_\mathcal{N}` |  bar over non-dominated points|
|$\bar{u}$| `\u` | `\bar{u}` |  Auxilary variable used in upper bound algorithm|
|$y^{ul}$| `\yul` | `y^{ul}` |  Upper-left lex min solution|
|$y^{lr}$| `\ylr` | `y^{lr}` |  Lower-right lex min solution|
|$\mathcal{Q}$| `\Q` | `\mathcal{Q}` |  Auxilary set used in upper bound algorithm|
## index sets
| Symbol |Shortcut|Command|Description| 
|-----|:------|:--------|:----------| 
|$\mathcal{I}$| `\I` | `\mathcal{I}` |  index set of decision variables|
|$\mathcal{J}$| `\J` | `\mathcal{J}` |  index set|
|$\mathcal{P}$| `\P` | `\mathcal{P}` |  index set objective space|
|$\mathcal{S}$| `\S` | `\mathcal{S}` |  set of subsystems|
|$\mathcal{Z}$| `\Z` | `\mathcal{Z}` |  set of subsystems|
## Number sets
| Symbol |Shortcut|Command|Description| 
|-----|:------|:--------|:----------| 
|$\mathbb{R}$| `\R` | `\mathbb{R}` |  The real numbers|
|$\mathbb{R}^p$| `\Rp` | `\mathbb{R}^p` |  The positive cone in p-dimensions|
|$\mathbb{R}_{\geqq}^p$| `\Rpp` | `\mathbb{R}_{\geqq}^p` |  The positive cone in p-dimensions including $0$|
|$\mathbb{R}_{\geq}^p$| `\Rppp` | `\mathbb{R}_{\geq}^p` |  The positive cone in p-dimensions without $0$|
|$\mathbb{N}$| `\N` | `\mathbb{N}` |  The set of natural numbes|
##  Operators
| Symbol |Shortcut|Command|Description| 
|-----|:------|:--------|:----------| 
|$\oplus$| `\+` | `\oplus` |  Minkowski sum|
|$\ominus$| `\-` | `\ominus` |  Minkowski difference|
|$\bigoplus$| `\msum` | `\bigoplus` |  Minkowski sum (multiple)|
|$\bigotimes$| `\mmul` | `\bigotimes` |  Cartesian product (multiple)|
|$\times$| `\x` | `\times` |  Cartesian product|
|$\prod$|  | `\prod` |  Cartesian product (multiple)|
|$(\cdot)_\mathcal{N}$|  | `(\cdot)_\mathcal{N}` |  Non-dominated points|
##  Misc
| Symbol |Shortcut|Command|Description| 
|-----|:------|:--------|:----------| 
|$e.g.\@\xspace$| `\eg` | `e.g.\@\xspace` | |
|$wrt.\@\xspace$| `\wrt` | `wrt.\@\xspace` | |
|$i.e.\@\xspace$| `\ie` | `i.e.\@\xspace` | |
|$branch-and-bound\@\xspace$| `\BB` | `branch-and-bound\@\xspace` | |
|$MO-MSP\@\xspace$| `\msp` | `MO-MSP\@\xspace` | |
|$multi-objective Minkowski Sum problem\@\xspace$| `\MSP` | `multi-objective Minkowski Sum problem\@\xspace` | |
|$\epsilon$-problem\@\xspace| `\eproblem` | `$\epsilon$-problem\@\xspace` | |
|$\epsilon$-method\@\xspace| `\emethod` | `$\epsilon$-method\@\xspace` | |

------

<!--END NOTATION-->
`README.md` file generated from tex file `LaTeXConfig/commands.tex` using the python script `generate_readme.py`