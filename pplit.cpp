#include "arg.h"
#include "chat.h"
#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static const char * MARK_PROMPT   = "<|@PROMPT@|>";
static const char * MARK_RESPONSE = "<|@RESPONSE@|>";
static const char * MARK_CONV     = "<|@CONV@|>";

struct ppl_turn {
    std::string prompt;
    std::string response;
};

using ppl_conv = std::vector<ppl_turn>;

struct row_stat {
    double nll;
    int    rank;
    int    arg;
    double logz;
};

struct totals {
    double           nll   = 0.0;
    int64_t          n     = 0;
    int64_t          top1  = 0;
    int64_t          top5  = 0;
    int64_t          top10 = 0;
    std::vector<int> ranks;
};

struct tally {
    int used    = 0;
    int skipped = 0;
};

struct kld_pair {
    int32_t id;
    float   logp;
};

struct kld_head {
    uint32_t magic;
    uint32_t version;
    uint32_t n_vocab;
    uint32_t topk;
    uint64_t rows;
};

struct kld_io {
    FILE *  out    = nullptr;
    FILE *  in     = nullptr;
    int     topk   = 64;
    int64_t torn   = 0;
    int64_t stored = 0;
    int64_t expect = 0;
};

struct kld_totals {
    double             sum  = 0.0;
    int64_t            n    = 0;
    int64_t            same = 0;
    std::vector<float> rows;
};

struct options {
    std::string save;
    std::string load;
    std::string tmpl;
    int         topk     = 64;
    int         cap      = 320;
    int         items    = 0;
    bool        selftest = false;
};

struct scorer {
    llama_context *               ctx;
    llama_batch *                 batch;
    const common_chat_templates * tmpls;
    int                           cap;
    int                           n_ctx;
    int                           n_batch;
    int                           n_vocab;
    kld_io *                      kld;
    kld_totals *                  kt;
};

static volatile int     g_item     = -1;
static volatile int     g_turn     = -1;
static volatile int     g_items    = 0;
static volatile int     g_ntok     = 0;
static volatile int     g_start    = 0;
static volatile int     g_chunkpos = -1;
static volatile int     g_chunksz  = 0;
static volatile int     g_nreq     = 0;
static volatile int64_t g_scored   = 0;
static const char *     g_model    = "?";

static void abort_hook(const char * msg) {
    fprintf(stderr,
            "\n[pplit] ggml aborted: %s\n"
            "[pplit]   model  %s\n"
            "[pplit]   conv   %d of %d, turn %d, %d tokens, reply at %d\n"
            "[pplit]   chunk  pos %d size %d, %d logit rows\n"
            "[pplit]   scored %lld tokens before this\n",
            msg ? msg : "(none)", g_model, g_item, g_items, g_turn,
            g_ntok, g_start, g_chunkpos, g_chunksz, g_nreq,
            (long long) g_scored);
    fflush(stderr);
}

static std::vector<char *> strip_opts(int argc, char ** argv, options & o) {
    std::vector<char *> keep;
    int i = 0;
    while (i < argc) {
        const bool has = i + 1 < argc;
        const bool sv  = has && std::strcmp(argv[i], "--kld-save") == 0;
        const bool ld  = has && std::strcmp(argv[i], "--kld") == 0;
        const bool tk  = has && std::strcmp(argv[i], "--kld-top") == 0;
        const bool cp  = has && std::strcmp(argv[i], "--cap") == 0;
        const bool it  = has && std::strcmp(argv[i], "--items") == 0;
        const bool tp  = has && std::strcmp(argv[i], "--template") == 0;
        const bool st  = std::strcmp(argv[i], "--selftest") == 0;
        if (sv) { o.save  = argv[i + 1]; }
        if (ld) { o.load  = argv[i + 1]; }
        if (tk) { o.topk  = std::atoi(argv[i + 1]); }
        if (cp) { o.cap   = std::atoi(argv[i + 1]); }
        if (it) { o.items = std::atoi(argv[i + 1]); }
        if (tp) { o.tmpl  = argv[i + 1]; }
        if (st) { o.selftest = true; }
        const bool pair = sv || ld || tk || cp || it || tp;
        if (!pair && !st) { keep.push_back(argv[i]); }
        i += pair ? 2 : 1;
    }
    return keep;
}

static std::string read_file(const char * path) {
    std::ifstream in(path);
    std::string out;
    if (in) {
        out.assign(std::istreambuf_iterator<char>(in),
                   std::istreambuf_iterator<char>());
    }
    return out;
}

static std::vector<std::string> split_lines(const std::string & text) {
    std::vector<std::string> lines;
    size_t p = 0;
    while (p <= text.size()) {
        const size_t e = text.find('\n', p);
        const size_t k = e == std::string::npos ? std::string::npos : e - p;
        lines.push_back(text.substr(p, k));
        p = e == std::string::npos ? text.size() + 1 : e + 1;
    }
    return lines;
}

static std::vector<ppl_conv> parse_corpus(const std::string & text) {
    const bool multi = text.find(std::string(MARK_CONV) + "\n") !=
                       std::string::npos;
    std::vector<ppl_conv> convs;
    ppl_conv cur;
    std::string prompt;
    std::string response;
    int where = 0;
    for (const auto & ln : split_lines(text)) {
        const bool have = !prompt.empty() && !response.empty();
        if (ln == MARK_CONV) {
            if (have) { cur.push_back({prompt, response}); }
            if (!cur.empty()) { convs.push_back(cur); }
            cur.clear();
            prompt.clear();
            response.clear();
            where = 0;
        } else if (ln == MARK_PROMPT) {
            if (have) { cur.push_back({prompt, response}); }
            if (have && !multi) {
                convs.push_back(cur);
                cur.clear();
            }
            prompt.clear();
            response.clear();
            where = 1;
        } else if (ln == MARK_RESPONSE) {
            where = 2;
        } else if (where == 1) {
            prompt += prompt.empty() ? ln : "\n" + ln;
        } else if (where == 2) {
            response += response.empty() ? ln : "\n" + ln;
        }
    }
    if (!prompt.empty() && !response.empty()) {
        cur.push_back({prompt, response});
    }
    if (!cur.empty()) { convs.push_back(cur); }
    return convs;
}

static row_stat score_row(const float * lg, int n_vocab, llama_token want) {
    const float lw = lg[want];
    float max_l = lg[0];
    int   arg   = 0;
    int   rank  = 0;
    for (int i = 0; i < n_vocab; ++i) {
        if (lg[i] > max_l) { max_l = lg[i]; arg = i; }
        if (lg[i] > lw)    { rank++; }
    }
    double sum = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum += std::exp((double) (lg[i] - max_l));
    }
    const double logz = (double) max_l + std::log(sum);
    return { logz - (double) lw, rank, arg, logz };
}

static std::string render(const common_chat_templates * tmpls,
                          const ppl_conv & conv, size_t upto) {
    common_chat_templates_inputs in;
    for (size_t i = 0; i <= upto; ++i) {
        common_chat_msg user;
        user.role    = "user";
        user.content = conv[i].prompt;
        in.messages.push_back(user);
        if (i < upto) {
            common_chat_msg asst;
            asst.role    = "assistant";
            asst.content = conv[i].response;
            in.messages.push_back(asst);
        }
    }
    in.add_generation_prompt = true;
    in.use_jinja             = true;
    in.enable_thinking       = false;
    in.add_bos               = false;
    std::string out;
    try {
        out = common_chat_templates_apply(tmpls, in).prompt;
    } catch (const std::exception & e) {
        LOG_ERR("%s: template failed: %s\n", __func__, e.what());
    }
    return out;
}

static bool kld_greater(const kld_pair & a, const kld_pair & b) {
    return a.logp > b.logp;
}

static void kld_scan(const float * lg, int n_vocab, double logz, int k,
                     std::vector<kld_pair> & top) {
    top.clear();
    for (int i = 0; i < n_vocab; ++i) {
        const float lp = (float) ((double) lg[i] - logz);
        if ((int) top.size() < k) {
            top.push_back({ i, lp });
            std::push_heap(top.begin(), top.end(), kld_greater);
        } else if (lp > top.front().logp) {
            std::pop_heap(top.begin(), top.end(), kld_greater);
            top.back() = { i, lp };
            std::push_heap(top.begin(), top.end(), kld_greater);
        }
    }
    std::sort_heap(top.begin(), top.end(), kld_greater);
}

static float kld_tail(const std::vector<kld_pair> & top) {
    double mass = 0.0;
    for (size_t i = 0; i < top.size(); ++i) {
        mass += std::exp((double) top[i].logp);
    }
    const double rest = mass < 1.0 ? 1.0 - mass : 0.0;
    return (float) std::log(rest > 1e-12 ? rest : 1e-12);
}

static void kld_write(kld_io & io, const std::vector<kld_pair> & top) {
    const float tail = kld_tail(top);
    io.stored += (int64_t) fwrite(top.data(), sizeof(kld_pair), top.size(),
                                  io.out);
    fwrite(&tail, sizeof(float), 1, io.out);
}

static float kld_read(kld_io & io, std::vector<kld_pair> & ref) {
    float tail = 0.0f;
    ref.resize((size_t) io.topk);
    const size_t got = fread(ref.data(), sizeof(kld_pair), ref.size(), io.in);
    const size_t gt  = fread(&tail, sizeof(float), 1, io.in);
    io.torn += (got == ref.size() && gt == 1) ? 0 : 1;
    return tail;
}

static void kld_row(const float * lg, double logz,
                    const std::vector<kld_pair> & ref, float tail, int arg,
                    kld_totals & kt) {
    double kl   = 0.0;
    double mass = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double pr = std::exp((double) ref[i].logp);
        const double qt = (double) lg[ref[i].id] - logz;
        kl   += pr * ((double) ref[i].logp - qt);
        mass += std::exp(qt);
    }
    const double rest = mass < 1.0 ? 1.0 - mass : 0.0;
    const double ptl  = std::exp((double) tail);
    kl += ptl * ((double) tail - std::log(rest > 1e-12 ? rest : 1e-12));
    kt.sum += kl;
    kt.rows.push_back((float) kl);
    kt.same += (!ref.empty() && ref[0].id == arg) ? 1 : 0;
    kt.n++;
}

static void fill_batch(llama_batch & batch,
                       const std::vector<llama_token> & toks, int lo,
                       int pos, int sz, std::vector<int> & want) {
    const int n = (int) toks.size();
    batch.n_tokens = 0;
    for (int k = 0; k < sz; ++k) {
        const int gi = pos + k;
        batch.token   [k]    = toks[gi];
        batch.pos     [k]    = gi;
        batch.n_seq_id[k]    = 1;
        batch.seq_id  [k][0] = 0;
        batch.logits  [k]    = gi >= lo && gi < n - 1;
        if (batch.logits[k] != 0) { want.push_back(gi); }
        batch.n_tokens++;
    }
}

static void accumulate(const scorer & s, const float * out,
                       const std::vector<int> & want,
                       const std::vector<llama_token> & toks, totals & acc) {
    std::vector<kld_pair> top;
    for (size_t j = 0; j < want.size(); ++j) {
        const float * lg = out + j * (size_t) s.n_vocab;
        const row_stat r = score_row(lg, s.n_vocab, toks[want[j] + 1]);
        acc.nll   += r.nll;
        acc.top1  += r.rank == 0 ? 1 : 0;
        acc.top5  += r.rank <  5 ? 1 : 0;
        acc.top10 += r.rank < 10 ? 1 : 0;
        acc.ranks.push_back(r.rank);
        acc.n++;
        if (s.kld != nullptr && s.kld->out != nullptr) {
            kld_scan(lg, s.n_vocab, r.logz, s.kld->topk, top);
            kld_write(*s.kld, top);
        }
        if (s.kld != nullptr && s.kld->in != nullptr) {
            const float tail = kld_read(*s.kld, top);
            kld_row(lg, r.logz, top, tail, r.arg, *s.kt);
        }
    }
    g_scored = acc.n;
}

static int score_chunk(const scorer & s,
                       const std::vector<llama_token> & toks, int lo,
                       int pos, int sz, totals & acc) {
    std::vector<int> want;
    fill_batch(*s.batch, toks, lo, pos, sz, want);
    g_chunkpos = pos;
    g_chunksz  = sz;
    g_nreq     = (int) want.size();
    const int rc = llama_decode(s.ctx, *s.batch);
    if (rc == 0) {
        accumulate(s, llama_get_logits(s.ctx), want, toks, acc);
    } else {
        LOG_ERR("%s: decode failed, item %d chunk %d\n", __func__, g_item,
                pos);
    }
    return rc == 0 ? sz : 0;
}

static int score_item(const scorer & s,
                      const std::vector<llama_token> & toks, int start,
                      totals & acc) {
    const int n  = (int) toks.size();
    const int lo = start - 1;
    int pos  = 0;
    int step = s.n_batch;
    llama_memory_clear(llama_get_memory(s.ctx), true);
    while (pos < n && step > 0) {
        step = score_chunk(s, toks, lo, pos,
                           std::min(s.n_batch, n - pos), acc);
        pos += step;
    }
    return pos;
}

static size_t score_turn(const scorer & s, const ppl_conv & conv, size_t t,
                         totals & acc, tally & count) {
    const std::string rendered = render(s.tmpls, conv, t);
    auto pid = common_tokenize(s.ctx, rendered, false, true);
    auto rid = common_tokenize(s.ctx, conv[t].response, false, false);
    if (s.cap > 0 && (int) rid.size() > s.cap) { rid.resize(s.cap); }
    std::vector<llama_token> toks = pid;
    toks.insert(toks.end(), rid.begin(), rid.end());
    size_t taken = 1;
    if (rendered.empty() || rid.size() < 2) {
        count.skipped++;
    } else if ((int) toks.size() > s.n_ctx) {
        LOG_WRN("%s: conv %d turn %zu needs %zu > n_ctx %d\n", __func__,
                g_item, t, toks.size(), s.n_ctx);
        count.skipped++;
    } else {
        g_ntok  = (int) toks.size();
        g_start = (int) pid.size();
        const int done = score_item(s, toks, (int) pid.size(), acc);
        taken = done >= (int) toks.size() ? 1 : 0;
        count.used += (int) taken;
    }
    return taken;
}

static size_t score_conv(const scorer & s, const ppl_conv & conv,
                         totals & acc, tally & count) {
    size_t t    = 0;
    size_t take = 1;
    while (t < conv.size() && take > 0) {
        g_turn = (int) t;
        take   = score_turn(s, conv, t, acc, count);
        t     += take;
    }
    return t;
}

static size_t score_items(const scorer & s,
                          const std::vector<ppl_conv> & convs,
                          totals & acc, tally & count) {
    size_t c    = 0;
    size_t take = 1;
    while (c < convs.size() && take > 0) {
        g_item = (int) c;
        take   = score_conv(s, convs[c], acc, count) >= convs[c].size();
        c     += take;
        if (acc.n > 0) {
            fprintf(stderr, "\r[pplit] %zu/%zu  ppl %.4f   ", c,
                    convs.size(), std::exp(acc.nll / (double) acc.n));
        }
    }
    fprintf(stderr, "\n");
    return c;
}

static void report(const char * path, totals & acc, int used, int skipped,
                   int convs) {
    std::sort(acc.ranks.begin(), acc.ranks.end());
    const double n    = (double) acc.n;
    const double last = (double) (acc.ranks.size() - 1);
    const char * slash = std::strrchr(path, '/');
    printf("%-30s ppl %8.4f  top1 %6.2f%%  top5 %6.2f%%  top10 %6.2f%%  "
           "rank p50 %d p90 %d   over %lld tokens, %d turns, %d convs",
           slash != nullptr ? slash + 1 : path, std::exp(acc.nll / n),
           100.0 * (double) acc.top1  / n,
           100.0 * (double) acc.top5  / n,
           100.0 * (double) acc.top10 / n,
           acc.ranks[(size_t) (0.50 * last)],
           acc.ranks[(size_t) (0.90 * last)],
           (long long) acc.n, used, convs);
    if (skipped > 0) {
        printf(", %d skipped", skipped);
    }
    printf("\n");
}

static bool kld_begin(kld_io & io, const options & o, int n_vocab) {
    kld_head h = { 0x444C4B50u, 1u, (uint32_t) n_vocab, (uint32_t) o.topk, 0u };
    if (!o.save.empty()) {
        io.out  = fopen(o.save.c_str(), "wb");
        io.topk = o.topk;
        if (io.out != nullptr) { fwrite(&h, sizeof(h), 1, io.out); }
    } else if (!o.load.empty()) {
        io.in = fopen(o.load.c_str(), "rb");
        h.magic = 0u;
        if (io.in != nullptr && fread(&h, sizeof(h), 1, io.in) == 1) {
            io.topk   = (int) h.topk;
            io.expect = (int64_t) h.rows;
        }
    }
    const bool bad_vocab = io.in != nullptr && (int) h.n_vocab != n_vocab;
    const bool no_file   = (!o.save.empty() && io.out == nullptr) ||
                           (!o.load.empty() && io.in == nullptr);
    if (no_file) {
        LOG_ERR("%s: cannot open the kld file\n", __func__);
    } else if (bad_vocab) {
        LOG_ERR("%s: kld vocab %u != model vocab %d\n", __func__,
                h.n_vocab, n_vocab);
    }
    return !no_file && !bad_vocab && h.magic == 0x444C4B50u;
}

static void kld_finish(kld_io & io, int64_t rows) {
    if (io.out != nullptr) {
        const uint64_t n = (uint64_t) rows;
        fseek(io.out, (long) (sizeof(uint32_t) * 4), SEEK_SET);
        fwrite(&n, sizeof(n), 1, io.out);
        fclose(io.out);
    }
    if (io.in != nullptr) { fclose(io.in); }
}

static void kld_report(kld_totals & kt, const kld_io & io) {
    std::sort(kt.rows.begin(), kt.rows.end());
    const double n    = (double) kt.n;
    const double last = (double) (kt.rows.size() - 1);
    printf("%-30s kld %8.5f  median %8.5f  p90 %8.5f  p99 %8.5f  "
           "top1-agree %6.2f%%   over %lld rows",
           "", kt.sum / n,
           (double) kt.rows[(size_t) (0.50 * last)],
           (double) kt.rows[(size_t) (0.90 * last)],
           (double) kt.rows[(size_t) (0.99 * last)],
           100.0 * (double) kt.same / n, (long long) kt.n);
    if (io.torn > 0) {
        printf(", %lld short reads", (long long) io.torn);
    }
    if (io.expect != kt.n) {
        printf(", ROW MISMATCH expected %lld", (long long) io.expect);
    }
    printf("\n");
}

static double kl_rows(const float * p, const float * q, int n) {
    double mp = p[0];
    double mq = q[0];
    for (int i = 1; i < n; ++i) {
        if (p[i] > mp) { mp = p[i]; }
        if (q[i] > mq) { mq = q[i]; }
    }
    double sp = 0.0;
    double sq = 0.0;
    for (int i = 0; i < n; ++i) {
        sp += std::exp((double) p[i] - mp);
        sq += std::exp((double) q[i] - mq);
    }
    const double lp = mp + std::log(sp);
    const double lq = mq + std::log(sq);
    double kl = 0.0;
    for (int i = 0; i < n; ++i) {
        const double a = (double) p[i] - lp;
        const double b = (double) q[i] - lq;
        kl += std::exp(a) * (a - b);
    }
    return kl;
}

static std::vector<float> probe(llama_model * model, common_params & params,
                                const std::vector<llama_token> & toks,
                                uint32_t ub, int stride) {
    llama_context_params cp = common_context_params_to_llama(params);
    cp.n_ctx    = (uint32_t) toks.size() + 8;
    cp.n_batch  = (uint32_t) toks.size();
    cp.n_ubatch = ub;
    llama_context * ctx = llama_init_from_model(model, cp);
    std::vector<float> out;
    if (ctx != nullptr) {
        const int n  = (int) toks.size();
        const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
        llama_batch batch = llama_batch_init(n, 0, 1);
        batch.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            batch.token   [i]    = toks[i];
            batch.pos     [i]    = i;
            batch.n_seq_id[i]    = 1;
            batch.seq_id  [i][0] = 0;
            batch.logits  [i]    = (int8_t) (i % stride == 0);
        }
        if (llama_decode(ctx, batch) == 0) {
            for (int i = 0; i < n; i += stride) {
                const float * lg = llama_get_logits_ith(ctx, i);
                if (lg != nullptr) { out.insert(out.end(), lg, lg + nv); }
            }
        }
        llama_batch_free(batch);
        llama_free(ctx);
    }
    return out;
}

static int selftest(llama_model * model, common_params & params,
                    const std::string & text) {
    const int stride = 16;
    const int nv     = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t ub = params.n_ubatch > 0 ? params.n_ubatch : 512;
    const size_t chars = std::min(text.size(), (size_t) ub * 12);
    const std::string head = text.substr(0, chars);
    std::vector<llama_token> toks((size_t) ub * 8);
    const int got = llama_tokenize(llama_model_get_vocab(model), head.c_str(),
                                   (int32_t) head.size(), toks.data(),
                                   (int32_t) toks.size(), true, false);
    const size_t want = (size_t) ub + (size_t) ub / 4;
    toks.resize(got > 0 ? std::min((size_t) got, want) : 0);
    int status = 1;
    if (toks.size() <= (size_t) ub) {
        LOG_ERR("%s: corpus too short to split at n_ubatch %u\n", __func__,
                ub);
    } else {
        const std::vector<float> a =
            probe(model, params, toks, (uint32_t) toks.size(), stride);
        const std::vector<float> b = probe(model, params, toks, ub, stride);
        if (a.size() != b.size() || a.empty()) {
            LOG_ERR("%s: probe produced no logits\n", __func__);
        } else {
            const int rows = (int) (a.size() / (size_t) nv);
            double sum  = 0.0;
            double worst = 0.0;
            int    diff  = 0;
            for (int r = 0; r < rows; ++r) {
                const float * pa = a.data() + (size_t) r * nv;
                const float * pb = b.data() + (size_t) r * nv;
                const double k = kl_rows(pa, pb, nv);
                sum += k;
                if (k > worst) { worst = k; }
                int ia = 0;
                int ib = 0;
                for (int i = 1; i < nv; ++i) {
                    if (pa[i] > pa[ia]) { ia = i; }
                    if (pb[i] > pb[ib]) { ib = i; }
                }
                diff += ia != ib ? 1 : 0;
            }
            const double mean = sum / (double) rows;
            const bool ok = mean < 1e-2;
            printf("selftest  %zu tokens, n_ubatch %u against %zu, "
                   "%d rows sampled\n", toks.size(), ub, toks.size(), rows);
            printf("selftest  argmax differs %d/%d, mean KL %.8f, "
                   "worst KL %.8f\n", diff, rows, mean, worst);
            printf("selftest  %s\n", ok
                   ? "PASS: splitting the batch does not change the "
                     "distribution"
                   : "FAIL: this backend changes results when the batch is "
                     "split; its numbers cannot be trusted");
            status = ok ? 0 : 2;
        }
    }
    return status;
}

static int score_all(common_params & params, const options & o,
                     const std::vector<ppl_conv> & items,
                     llama_model * model, llama_context * ctx) {
    std::string tmpl = !o.tmpl.empty() ? read_file(o.tmpl.c_str())
                                       : params.chat_template;
    auto tmpls = common_chat_templates_init(model, tmpl);
    LOG_INF("%s: template %zu bytes%s\n", __func__,
            common_chat_templates_source(tmpls.get()).size(),
            tmpl.empty() ? " (from the GGUF)" : " (overridden)");
    LOG_INF("%s: n_ctx %u n_batch %u n_ubatch %u cap %d\n", __func__,
            llama_n_ctx(ctx), llama_n_batch(ctx), llama_n_ubatch(ctx),
            o.cap);
    llama_batch batch = llama_batch_init(params.n_batch, 0, 1);
    const int  n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const bool want_kld = !o.save.empty() || !o.load.empty();
    kld_io     io;
    kld_totals kt;
    const bool kld_ok = !want_kld || kld_begin(io, o, n_vocab);
    const scorer s = {
        ctx, &batch, tmpls.get(), o.cap,
        (int) llama_n_ctx(ctx), (int) params.n_batch, n_vocab,
        want_kld ? &io : nullptr, &kt
    };
    totals acc;
    tally  count;
    const size_t done = kld_ok ? score_items(s, items, acc, count) : 0;
    const bool complete = kld_ok && done >= items.size() && acc.n > 0;
    if (complete) {
        report(params.model.path.c_str(), acc, count.used, count.skipped,
               (int) items.size());
    } else if (kld_ok && done >= items.size()) {
        LOG_ERR("%s: nothing scored\n", __func__);
    }
    if (complete && kt.n > 0) {
        kld_report(kt, io);
    }
    if (want_kld) {
        kld_finish(io, acc.n);
    }
    llama_batch_free(batch);
    return complete ? 0 : 1;
}

static int run(common_params & params, const options & o,
               const std::vector<ppl_conv> & items) {
    llama_backend_init();
    llama_numa_init(params.numa);
    auto init = common_init_from_params(params);
    auto * model = init->model();
    auto * ctx   = init->context();
    int status = 1;
    if (model != nullptr && ctx != nullptr && o.selftest) {
        status = selftest(model, params, params.prompt);
    } else if (model != nullptr && ctx != nullptr) {
        status = score_all(params, o, items, model, ctx);
    } else {
        LOG_ERR("%s: failed to load the model\n", __func__);
    }
    llama_backend_free();
    return status;
}

int main(int argc, char ** argv) {
    common_params params;
    options       o;
    params.n_ctx  = 8192;
    params.escape = false;
    int status = 0;
    common_init();
    ggml_set_abort_callback(abort_hook);
    std::vector<char *> av = strip_opts(argc, argv, o);
    if (!common_params_parse((int) av.size(), av.data(), params,
                             LLAMA_EXAMPLE_PERPLEXITY)) {
        status = 1;
    } else if (params.prompt.empty()) {
        LOG_ERR("%s: give the corpus with -f <corpus.txt>\n", __func__);
        status = 1;
    } else {
        auto items = parse_corpus(params.prompt);
        if (o.items > 0 && (int) items.size() > o.items) {
            items.resize((size_t) o.items);
        }
        if (items.empty()) {
            LOG_ERR("%s: no items; are %s / %s alone on their lines?\n",
                    __func__, MARK_PROMPT, MARK_RESPONSE);
            status = 1;
        } else {
            g_model = params.model.path.c_str();
            g_items = (int) items.size();
            LOG_INF("%s: %zu items\n", __func__, items.size());
            status = run(params, o, items);
        }
    }
    return status;
}
