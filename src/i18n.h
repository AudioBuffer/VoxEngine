#pragma once
// 轻量双语支持：中文 / English
//   T("中文","English")  —— 按当前语言选取
//   语言偏好持久化到 ~/.voxengine_lang
#include <cstdlib>
#include <string>
#include <fstream>

namespace vox {

inline int& langSlot() { static int l = -1; return l; }

inline const char* langFilePath() {
    static std::string p;
    if (p.empty()) {
        const char* h = std::getenv("HOME");
        p = std::string(h ? h : ".") + "/.voxengine_lang";
    }
    return p.c_str();
}

inline void loadLang() {
    if (langSlot() >= 0) return;
    langSlot() = 0;                                   // 默认中文
    std::ifstream f(langFilePath());
    std::string s;
    if (f >> s) langSlot() = (s == "en") ? 1 : 0;
}

inline void saveLang(int l) {
    langSlot() = l;
    std::ofstream f(langFilePath());
    if (f) f << (l == 1 ? "en" : "zh") << "\n";
}

inline int  lang() { loadLang(); return langSlot(); }
inline bool isEn() { return lang() == 1; }

// 按当前语言取串
inline const char* T(const char* zh, const char* en) { return isEn() ? en : zh; }

} // namespace vox
