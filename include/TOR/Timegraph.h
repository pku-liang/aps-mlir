#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <stack>
#include <vector>

class Draw {
private:
  int cnt, cnts;

public:
  std::ofstream outFile;
  Draw();
  int Create_node(std::string nod);
  int Create_node(int nod);
  int Create_edge(std::string st, std::string ed, std::string mg);
  int Create_edge(std::string st, std::string ed, std::string mg,
                  std::string clo);
  void End();
  void out_png(std::string fp);
};

class make_graph {
private:
  struct gp {
    int n, to;
    std::string mg;
  };
  std::string s;
  gp *edge;
  bool sim = false, lay = false; // sim控制是否简化生成的图形
  int *head;
  int cnt, mas, len, p, lif; // lif在stoint中对if进行额外建边
  std::string filename;
  std::map<int, int> cns;
  std::string edge_color[21] = {
      "forestgreen",    "black",      "#008B8B",       "brown1",
      "darkgoldenrod4", "indigo",     "indianred",     "chocolate",
      "firebrick1",     "violetred4", "darklawngreen", "magenta4",
      "midnightblue",   "darkcoral",  "#008B8B",       "darkgoldenrodyellow",
      "darkgray",       "green",      "darkgrey",      "forestgreen"};
  std::vector<std::pair<int, int>> col;
  std::map<int, std::map<int, int>> dedup;
  std::istringstream iss;
  void init();          // 进行字符串的解析
  inline void resize(); // 对当前数组进行动态扩容
  inline void add(int x, int y, std::string ms); // 利用链式前向星建边
  inline void stoint(std::string s, int &st, int &ed); // 把字符串转为数字
public:
  make_graph(std::string dumpStr);
  ~make_graph();
  void out(std::string furl);
  void out();
  void simplify_graph();
  void lr_layout();
};

Draw::Draw() { cnt = 0, cnts = 0; }

int Draw::Create_node(std::string nod) {
  //     strtoint[nod] = ++cnt;
  //     inttostr[cnt] = nod;
  outFile << nod << ' ';
  if (cnt & 1)
    outFile << "[color=blue];" << '\n';
  else
    outFile << "[color=red];" << '\n';
  return ++cnt;
}

int Draw::Create_node(int nod) {
  std::string s1 = std::to_string(nod);
  return Create_node(s1);
}

int Draw::Create_edge(std::string st, std::string ed, std::string mg,
                      std::string clo) {
  outFile << st << " -> " << ed << " [label=\"" << mg
          << "\",color = \"" + clo + "\"] ;" << '\n';
  return ++cnts;
}

int Draw::Create_edge(std::string st, std::string ed, std::string mg) {
  outFile << st << " -> " << ed << " [label=\"" << mg << "\"];" << '\n';
  return ++cnts;
}

void Draw::End() { outFile << '}'; }

void Draw::out_png(std::string fp) {
  std::string command = "dot -Tpng -Gdpi=300 ./tn.dot -o ";
  command += fp;
  std::system(command.c_str());
}

inline void make_graph::resize() { // 对当前数组进行动态扩容
  gp *temp;
  int *temh;
  try {
    temp = new gp[len * 2];
    temh = new int[len * 2];
  } catch (const std::bad_alloc &e) {
    std::cerr << "内存分配失败 " << e.what() << '\n';
  }
  for (int i = 0; i < len; ++i) {
    temp[i] = edge[i];
    temh[i] = head[i];
    temh[i + len] = 0;
  }
  len *= 2;
  delete[] edge;
  delete[] head;
  //  std::cout<<"resize  "<<len/2<<" -> "<<len<<'\n';
  // std::cout<<mas<<"   "<<cnt<<'\n';
  edge = temp;
  head = temh;
}

inline void make_graph::add(int x, int y,
                            std::string ms) { // 利用链式前向星建边
  if (x == y)
    return;
  mas = std::max(mas, std::max(x, y));
  while (mas >= len || ++cnt >= len)
    resize();
  cns[x] = 1, cns[y] = 1;
  if (ms != "")
    dedup[x][y]++, dedup[y][x]++;
  edge[cnt].to = y;
  edge[cnt].n = head[x];
  edge[cnt].mg = ms;
  head[x] = cnt;
  p = y;
  //  std::cout<<x<<"  "<<y<<"  "<<s<<'\n';
}

inline void make_graph::stoint(std::string s, int &st,
                               int &ed) { // 把字符串转为数字
  int j = s.size() - 1, m = 1;
  while (1) {
    while (j > 0 && s[j] != ')')
      j--;
    if (j == 0)
      return;
    j--;
    int tem = j - 1;
    while (s[tem] != 't') {
      if (s[tem] == ')')
        j = tem - 1;
      tem--;
    }
    if (s[tem - 1] == ' ' && s[tem + 1] == 'o' && s[tem + 2] == ' ')
      break;
    else
      j = tem;
  }

  for (; s[j] != ' '; j--) {
    ed += (s[j] - '0') * m;
    m *= 10;
  }
  j -= 4, m = 1;
  for (; s[j] != '('; j--) {
    st += (s[j] - '0') * m;
    m *= 10;
  }
  if (lif > 0) {
    add(lif, st, "if");
    lif = -lif;
  }
}

make_graph::make_graph(std::string dumpStr) {
  iss = std::istringstream(dumpStr);
  edge = new gp[10001];
  head = new int[10001];
  for (int i = 0; i < 10001; ++i) {
    head[i] = 0;
  }
  cnt = 0, mas = 0, len = 10001, lif = 0;
  init();
}

make_graph::~make_graph() {
  delete[] edge;
  delete[] head;
}

void make_graph::out(std::string furl) {
  Draw ppp;
  std::string dotPath = "tn.dot";
  ppp.outFile = std::ofstream(dotPath);
  furl += "/timegraph.png";
  // 输出点和点之间的关系
  if (furl.size() <= 4 || furl[furl.size() - 4] != '.') {
    furl += filename;
    furl += ".png";
  }
  //    std::cout<<furl<<std::endl;

  // std::cout<<"  _background=\"c 7 -#ff0000 p 4 4 4 36 4 36 36 4 36\";"<<"\n";

  ppp.outFile << "digraph{\n";
  if (lay)
    ppp.outFile << "rankdir=LR;" << '\n';
  for (auto xx : cns) {
    if (xx.second == 1 && xx.first < 2e6)
      ppp.Create_node(xx.first);
  }
  if (sim) {
    struct nos {
      int i, j;
      std::string mg;
      bool operator<(const nos &other) const {
        if (i != other.i)
          return i < other.i;
        if (j != other.j)
          return j < other.j;
        return mg < other.mg;
      }
    };
    std::map<nos, int> noo;
    nos temp;
    for (int i = 0; i < mas; ++i) {
      for (int j = head[i]; j; j = edge[j].n) {
        if (edge[j].mg == "" && dedup[i][edge[j].to] > 0)
          continue;
        temp.i = i;
        temp.j = edge[j].to;
        temp.mg = edge[j].mg;
        noo[temp]++;
      }
    }
    for (auto &x : noo) {
      ppp.outFile << x.second << '\n';
      int tnc = 0;
      int ti = x.first.i, tj = x.first.j;
      if (ti > tj)
        std::swap(ti, tj);
      for (auto y : col) {
        if (ti >= y.first && tj <= y.second) {
          tnc++;
        }
        //  std::cout<<y.first<<"  "<<y.second<<'\n';
      }

      if (x.second > 1)
        ppp.Create_edge(std::to_string(x.first.i), std::to_string(x.first.j),
                        x.first.mg + " : " + std::to_string(x.second),
                        edge_color[tnc % 20]);
      else if (x.second == 1)
        ppp.Create_edge(std::to_string(x.first.i), std::to_string(x.first.j),
                        x.first.mg, edge_color[tnc % 20]);
    }
  } else {
    struct nos {
      std::string i, j, mg;
      int tnc;
    };
    std::stack<nos> noo;
    nos temp;
    for (int i = 0; i <= mas; ++i) {
      for (int j = head[i]; j; j = edge[j].n) {
        if (edge[j].mg == "" && dedup[i][edge[j].to] > 0)
          continue;
        temp.i = std::to_string(i);
        temp.j = std::to_string(edge[j].to);
        temp.mg = edge[j].mg;
        temp.tnc = 0;
        int ti = i, tj = edge[j].to;
        if (ti > tj)
          std::swap(ti, tj);
        for (auto y : col) {

          if (ti >= y.first && tj <= y.second)
            temp.tnc++;
          //  std::cout<<y.first<<"  "<<y.second<<'\n';
        }
        noo.emplace(temp);
      }
    }
    while (!noo.empty()) {
      ppp.Create_edge(noo.top().i, noo.top().j, noo.top().mg,
                      edge_color[noo.top().tnc % 20]);
      noo.pop();
    }
  }

  ppp.End();
  ppp.outFile.close();
  ppp.out_png(furl);
  remove(dotPath.c_str());
}

void make_graph::out() { // 输出点和点之间的关系
  make_graph::out("");
}

// 用来简化图中的重边，如果此时有x条A -> B的addi边，那么会缩写成addi : x；
void make_graph::simplify_graph() { sim = true; }

void make_graph::lr_layout() { lay = true; }

void make_graph::init() { // 进行字符串的解析
  // std::stack<std::pair<int,int> >q;
  while (getline(iss, s)) {
    // std::cout << s << "\n";
    std::string s1 = "";
    int el = 0, cs = 0; // cs表示括号数量
    for (int i = 0, len = s.size(); i < len;
         ++i) { // s1在这里存储 . 后面的字符.比如tor.muli,在读取 .
                // 后,会把后面的字符全部存储到s1中,直到空格为止.
      if (s[i] == '.' && s[i - 1] == 'r' && s[i - 2] == 'o' && s1.empty()) {
        for (int j = 1; i + j < len && s[i + j] != ' ' && s[i + j] != '<'; ++j) {
          s1 += s[i + j];
        }
      } else if (len - i > 3 && s[i] == 'e' && s[i + 2] == 's' &&
                 s[i + 3] == 'e') {
        el = 1;
      } else if (s[i] == '{')
        cs++;
      else if (s[i] == '}')
        cs--;
    }

    if ((cs < 0 && s1.empty()) || (cs == 0 && el))
      s1 = '}';
    //  std::cout<<s<<"   "<<s1<<'\n';
    if (s1 == "for" ||
        s1 ==
            "while") { // 如果是for或者while，on会换行，这里把下一行读取，然后合并到当前行
      std::string s2;
      getline(iss, s2);
      s += s2;
    } else if (s1 == "constant" || s1 == "alloc" || s1 == "succ" || s1 == "" ||
               s1 == "memref")
      continue;
    if (s1 == "func" || s1 == "design" || s1 == "module") {
      if (s1 == "design") {
        int bol = 0;
        for (int i = 0, len = s.size(); i < len; ++i) {
          if (bol && s[i] == ' ')
            break;
          if (bol)
            filename += s[i];
          if (s[i] == '@')
            bol = 1;
        }
      }
      continue;
    }
    if (s1 == "timegraph") {
      int st = 0, ed = 0;
      stoint(s, st, ed);
      add(st, ed, "");
      col.push_back({st, ed});
      std::string ss1;
      while (getline(iss, ss1)) {
        if (ss1[ss1.size() - 1] == '}')
          break;
        st = 0, ed = 0;
        int i = 1;
        while (ss1[i] < '0' || ss1[i] > '9')
          i++;

        for (; ss1[i] != ' '; ++i) {
          st *= 10;
          st += ss1[i] - '0';
        }

        while (ss1[i] < '0' || ss1[i] > '9')
          i++;
        for (; ss1[i] != ' '; ++i) {
          ed *= 10;
          ed += ss1[i] - '0';
        }
        if (ss1[ss1.size() - 6] == 'f')
          add(ed, st, "loop exit");
        else if (ss1[ss1.size() - 8] == 'w')
          add(ed, st, "loop exit(while)");
        else
          add(ed, st, "");
        while (ss1[i] != ']') {
          if (ss1[i] == ',') {
            ed = 0;
            i += 2;
            //      std::cout<<ss1<<'\n';
            for (; ss1[i] != ' '; i++) {
              ed = ed * 10 + ss1[i] - '0';
            }
            //   std::cout<<ed<<'\n';
            add(ed, st, "");
          }
          i++;
        }
      }
      continue;
    }
    if (el)
      lif = abs(
          lif); // 如果是else的情况，那么if中和else中最后的一个元素都需要指向on后面的目标元素。所以在这里需要把栈顶的元素，也就是当前else对应的目标元素重复入栈。
    if (s1 == "for" || s1 == "if" || s1 == "while") {
      int st = 0, ed = 0;
      stoint(s, st, ed);
      if (s1 == "for") {
        //       std::cout<<p<<' '<<st<<' '<<q.size()<<'\n';
        //      if (q.size() > 4)add(p,st,"loop");
        col.push_back({st, ed});
        add(ed, st, "loop back");
      } else if (s1 == "if")
        lif = st;
      else {
        col.push_back({st, ed});
        add(ed, st, "loop back(while)");
      }
      continue;
    } else if (s1 == "}" && !el) {
      // std::cout<<el<<'\n';
      lif = -abs(lif);
      continue;
    }
    if (s1 == "yield")
      continue; // 直接跳过,对点没有影响
    int st = 0, ed = 0;
    stoint(s, st, ed);
    // std::cout<<st<<"  "<<ed<<"  "<<lif<<'\n';
    add(st, ed, s1);
    s.clear();
  }
}
