#pragma once
// 音素总表：汉语拼音 / 英语(ARPABET) / 日语罗马字
// 统一映射到引擎基元： vowel 0=a 1=o 2=e 3=i 4=u 5=ü(-1=无元音)
//                      cons  0=无 1=S 2=Sh 3=F 4=P 5=B 6=M
#include <string>
#include <cctype>
#include <algorithm>

namespace vox {

struct PhonemeDef { const char* sym; int vowel; int cons; };

inline const PhonemeDef kPhonemes[] = {
    // ================= 汉语拼音 =================
    {"b",-1,4},{"p",-1,4},{"m",-1,6},{"f",-1,3},
    {"d",-1,4},{"t",-1,4},{"n",-1,6},{"l",-1,5},
    {"g",-1,4},{"k",-1,4},{"h",-1,3},
    {"j",-1,2},{"q",-1,2},{"x",-1,2},
    {"zh",-1,2},{"ch",-1,2},{"sh",-1,2},{"r",-1,5},
    {"z",-1,1},{"c",-1,1},{"s",-1,1},
    {"a",0,0},{"o",1,0},{"e",2,0},{"i",3,0},{"u",4,0},{"v",5,0},
    {"ai",0,0},{"ei",2,0},{"ao",0,0},{"ou",1,0},{"er",2,0},
    {"ia",0,0},{"ie",2,0},{"iao",0,0},{"iu",1,0},{"ua",0,0},{"uo",1,0},{"ui",2,0},
    {"an",0,0},{"en",2,0},{"in",3,0},{"un",4,0},{"vn",5,0},
    {"ang",0,0},{"eng",2,0},{"ing",3,0},{"ong",1,0},
    // ================= 英语 ARPABET =================
    {"aa",0,0},{"ae",0,0},{"ah",1,0},{"aw",1,0},{"ay",0,0},
    {"eh",2,0},{"ey",2,0},{"ih",3,0},{"iy",3,0},
    {"ow",1,0},{"oy",1,0},{"uh",4,0},{"uw",4,0},
    {"jh",-1,2},{"zh",-1,2},{"dh",-1,3},{"th",-1,3},{"v",-1,3},
    {"hh",-1,3},{"ng",-1,6},
    // ================= 日语罗马字 =================
    {"ts",-1,1},{"nn",-1,6},{"q",-1,4},{"N",-1,6},{"j",-1,2},
    // 介音（中/日/英通用）
    {"w",-1,5},{"y",-1,5},
};

inline std::string phNorm(const std::string& raw) {
    std::string t;
    for (size_t i = 0; i < raw.size();) {
        const unsigned char c = (unsigned char)raw[i];
        if (c == 0xC3 && i + 1 < raw.size()) { t += 'v'; i += 2; continue; }  // UTF-8 ü
        t += (char)std::tolower(c); ++i;
    }
    for (size_t p = t.find("u:"); p != std::string::npos; p = t.find("u:"))
        t.replace(p, 2, "v");
    return t;
}

inline const char* vowelName(int v) {
    static const char* n[6] = {"a","o","e","i","u","ü"};
    return (v >= 0 && v < 6) ? n[v] : "-";
}

// 主查表：先整体匹配，再按「声母 + 韵母」最长前缀拆分（覆盖中/日 CV 音节，如 ni / hao / ka）
inline bool lookupPhonemeAll(const std::string& raw, int& vowel, int& cons) {
    const std::string t = phNorm(raw);
    if (t.empty()) return false;
    for (const auto& p : kPhonemes)
        if (t == p.sym) { vowel = p.vowel; cons = p.cons; return true; }
    const int L = (int)std::min<size_t>(3, t.size());
    for (int n = L; n >= 1; --n) {
        if (n >= (int)t.size()) continue;
        const std::string a = t.substr(0, (size_t)n), b = t.substr((size_t)n);
        int ca = -1, vb = -1;
        bool oka = false, okb = false;
        for (const auto& p : kPhonemes) {
            if (a == p.sym && p.vowel < 0) { ca = p.cons; oka = true; }
            if (b == p.sym && p.cons == 0 && p.vowel >= 0) { vb = p.vowel; okb = true; }
        }
        if (oka && okb) { vowel = vb; cons = ca; return true; }
    }
    return false;
}

} // namespace vox
