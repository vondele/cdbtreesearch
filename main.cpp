#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string_view>
#include <unistd.h>

#include "external/chess.hpp"

#include "cdbdirect.h"

using namespace chess;

// get memory in MB
std::pair<size_t, size_t> get_memory() {
  size_t tSize = 0, resident = 0, share = 0;
  std::ifstream buffer("/proc/self/statm");
  buffer >> tSize >> resident;
  buffer.close();

  long page_size = sysconf(_SC_PAGE_SIZE);
  return std::make_pair(tSize * page_size / (1024 * 1024),
                        resident * page_size / (1024 * 1024));
};

// locate command line arguments
inline bool find_argument(const std::vector<std::string> &args,
                          std::vector<std::string>::const_iterator &pos,
                          std::string_view arg,
                          bool without_parameter = false) {
  pos = std::find(args.begin(), args.end(), arg);

  return pos != args.end() &&
         (without_parameter || std::next(pos) != args.end());
}

// local data and time
std::string getCurrentDateTime() {
  // Get current time
  std::time_t now = std::time(nullptr);

  // Convert it to local time structure
  std::tm *localTime = std::localtime(&now);

  // Create a stringstream to format the date and time
  std::ostringstream dateTimeStream;
  dateTimeStream << (1900 + localTime->tm_year) << "-"
                 << (localTime->tm_mon + 1) << "-" << localTime->tm_mday << " "
                 << localTime->tm_hour << ":" << localTime->tm_min << ":"
                 << localTime->tm_sec;

  // Convert to string and return
  return dateTimeStream.str();
};

struct Stats {
  std::atomic<size_t> gets;
  std::atomic<size_t> hits;
  std::atomic<size_t> nodes;

  void clear() { gets = hits = nodes = 0; };
};

using score_t = int;
using depth_t = int;

using search_result_t = std::pair<score_t, depth_t>;

search_result_t gameOver(Board &board, int depth) {

  // game ends, no attempt for now to delay mate as long as possible, i.e. all
  // mates scored the same.
  auto res = board.isGameOver();
  if (res.second == GameResult::LOSE)
    return std::make_pair(-30000, depth);

  if (res.second == GameResult::DRAW)
    return std::make_pair(0, depth);

  return std::make_pair(std::numeric_limits<int>::min(), -1);
}

// for chess960 uci::uciToMove only accepts KxR castling notation
Move cdbuci_to_move(const Board &board, std::string_view uci) {
  if (board.chess960() &&
      (uci == "e1g1" || uci == "e1c1" || uci == "e8g8" || uci == "e8c8")) {
    auto source = Square(uci.substr(0, 2));
    auto target = Square(uci.substr(2, 2));
    auto pt = board.at(source).type();

    if (pt == PieceType::KING && board.at(target).type() == PieceType::NONE) {
      target =
          Square(target > source ? File::FILE_H : File::FILE_A, source.rank());
      return Move::make<Move::CASTLING>(source, target);
    }
  }
  return uci::uciToMove(board, uci);
}

search_result_t pvsearch(Board &board, int ply, Stats &stats,
                         const std::uintptr_t handle) {

  stats.nodes++;

  search_result_t ret = gameOver(board, ply);
  if (ret.second >= 0) {
    std::cout << "Ply " << std::setw(4) << ply << " Game over " << std::setw(5)
              << (ply % 2 == 0 ? 1 : -1) * ret.first << std::endl;
    return ret;
  }

  // probe DB
  stats.gets++;
  std::vector<std::pair<std::string, int>> result =
      cdbdirect_get(handle, board.getXfen(false));
  size_t n_elements = result.size();
  int rply = result[n_elements - 1].second;

  if (rply == -2 || n_elements <= 1)
    return std::make_pair(0, -1);

  stats.hits++;

  std::cout << "ply " << std::setw(4) << ply << " bestmove " << std::setw(5)
            << result[0].first << " score " << std::setw(6)
            << (ply % 2 == 0 ? 1 : -1) * result[0].second << std::endl;

  Move m = cdbuci_to_move(board, result[0].first);
  board.makeMove<true>(m);
  search_result_t sr = pvsearch(board, ply + 1, stats, handle);
  if (sr.second <= 0)
    sr = std::make_pair(result[0].second, ply);
  board.unmakeMove(m);
  return sr;
}

search_result_t minimax(Board &board, int depth, Stats &stats,
                        const std::uintptr_t handle) {

  stats.nodes++;
  search_result_t ret = gameOver(board, depth);
  if (ret.second >= 0)
    return ret;

  // probe DB
  stats.gets++;
  std::vector<std::pair<std::string, int>> result =
      cdbdirect_get(handle, board.getXfen(false));
  size_t n_elements = result.size();
  int ply = result[n_elements - 1].second;

  if (ply == -2 || n_elements <= 1) {
    return std::make_pair(0, -1);
  }

  stats.hits++;

  if (depth == 0)
    return std::make_pair(result[0].second, 0);

  // Now explore the remaining moves
  for (auto &pair : result)
    if (pair.first != "a0a0") {

      Move m = cdbuci_to_move(board, pair.first);
      board.makeMove<true>(m);
      search_result_t moveresult = minimax(board, depth - 1, stats, handle);
      board.unmakeMove(m);

      // Not in DB, use scored move result
      if (moveresult.second < 0)
        moveresult = std::make_pair(-pair.second, depth);

      // better score ?
      if (moveresult.second >= 0 && -moveresult.first >= ret.first) {
        ret = std::make_pair(-moveresult.first, moveresult.second);
      }
    }

  // None of the scored moves led to position in the db, return bestmove.
  if (ret.second < 0)
    ret = std::make_pair(result[0].second, depth);

  return ret;
}

search_result_t alphabeta(Board &board, int depth, Stats &stats,
                          const std::uintptr_t handle, int alpha, int beta,
                          std::int64_t margin) {

  stats.nodes++;
  search_result_t ret = gameOver(board, depth);
  if (ret.second >= 0)
    return ret;

  // probe DB
  stats.gets++;
  std::vector<std::pair<std::string, int>> result =
      cdbdirect_get(handle, board.getXfen(false));
  size_t n_elements = result.size();
  int ply = result[n_elements - 1].second;

  if (ply == -2 || n_elements <= 1)
    return std::make_pair(42, -1);

  stats.hits++;

  if (depth == 0)
    return std::make_pair(result[0].second, 0);

  // Now explore the remaining moves
  for (auto &pair : result)
    if (pair.first != "a0a0") {

      // we trust the score no to exceed alpha, can thus have no effect on
      // search
      if (pair.second < alpha - margin * (depth + 1)) {
        continue;
      }

      // if we trust the score to exceed beta, we won't search and fail high.
      search_result_t moveresult;
      if (pair.second > beta + margin * (depth + 1)) {
        moveresult = std::make_pair(-pair.second, depth);
      } else {
        Move m = cdbuci_to_move(board, pair.first);
        board.makeMove<true>(m);
        moveresult =
            alphabeta(board, depth - 1, stats, handle, -beta, -alpha, margin);
        board.unmakeMove(m);

        // Not in DB, use scored move result
        if (moveresult.second < 0) {
          moveresult = std::make_pair(-pair.second, depth);
        }
      }

      // better score ?
      if (-moveresult.first >= ret.first) {
        ret = std::make_pair(-moveresult.first, moveresult.second);

        if (ret.first >= beta)
          break;

        alpha = ret.first;
      }
    }

  return ret;
}

int main(int argc, char const *argv[]) {

  const std::vector<std::string> args(argv + 1, argv + argc);
  std::vector<std::string>::const_iterator pos;

  // starting fen, defaults to 1. g4
  std::string fen =
      "rnbqkbnr/pppppppp/8/8/6P1/8/PPPPPP1P/RNBQKBNR b KQkq - 0 1";
  if (find_argument(args, pos, "--fen"))
    fen = {*std::next(pos)};

  if (fen == "startpos")
    fen = constants::STARTPOS;

  bool chess960 = find_argument(args, pos, "--chess960", true);

  // (initial) depth for searches
  int depth = 0;
  if (find_argument(args, pos, "--depth"))
    depth = std::stoi(*std::next(pos));

  // how much to trust cdb move scores in alpha beta search
  // only search a move if it is in ]alpha - margin * (depth + 1), beta + margin
  // * (depth + 1)[ the default of 5 is probably rather accurate, but with a
  // value of 50k all moves (max score 30k) will always be searched.
  std::int64_t margin = 5;
  if (find_argument(args, pos, "--margin"))
    margin = std::stoi(*std::next(pos));

  bool do_minimax = find_argument(args, pos, "--minimax", true);
  bool do_alphabeta = find_argument(args, pos, "--alphabeta", true);
  bool do_pvsearch = find_argument(args, pos, "--pvsearch", true);
  bool allmoves = find_argument(args, pos, "--moves", true);

  Board board(fen, chess960);

  std::cout << "Search fen: " << board.getXfen(false) << std::endl;
  std::cout << "     depth: " << depth << std::endl;

  std::uintptr_t handle = cdbdirect_initialize(CHESSDB_PATH);
  std::uint64_t db_size = cdbdirect_size(handle);
  std::cout << "DB count: " << db_size << std::endl;

  Stats stats;
  if (do_pvsearch) {
    std::cout << std::endl << "Starting PV search... " << std::endl;
    stats.clear();
    search_result_t sr = pvsearch(board, 0, stats, handle);
    std::cout << "         "
              << " nodes:" << std::setw(12) << stats.nodes << std::endl
              << "         "
              << " gets: " << std::setw(12) << stats.gets << std::endl
              << "         "
              << " hits: " << std::setw(12) << stats.hits << std::endl;
  }

  // minimax for debugging reference
  if (do_minimax) {
    std::cout << std::endl << "Starting minimax search... " << std::endl;
    stats.clear();
    search_result_t sr = minimax(board, depth, stats, handle);
    std::cout << "At depth " << std::setw(4) << depth << " found score "
              << std::setw(6) << -sr.first << " from a position at "
              << std::setw(4) << depth - sr.second << " plies from root"
              << std::endl;
    std::cout << "         "
              << " nodes:" << std::setw(12) << stats.nodes
              << " bf: " << std::setw(12)
              << std::exp(std::log((double)stats.nodes) / std::max(1, depth))
              << std::endl
              << "         "
              << " gets: " << std::setw(12) << stats.gets
              << " bf: " << std::setw(12)
              << std::exp(std::log((double)stats.gets) / std::max(1, depth))
              << std::endl
              << "         "
              << " hits: " << std::setw(12) << stats.hits
              << " bf: " << std::setw(12)
              << std::exp(std::log((double)stats.hits) / std::max(1, depth))
              << std::endl;
  }

  if (do_alphabeta) {

    std::cout << std::endl << "Starting alphabeta search... " << std::endl;

    auto anaboard = [&depth, &stats, &handle, &margin](Board &board) {
      int alpha = -30001;
      int beta = 30001;
      stats.clear();
      auto t_start = std::chrono::high_resolution_clock::now();
      search_result_t sr =
          alphabeta(board, depth, stats, handle, alpha, beta, margin);
      auto t_end = std::chrono::high_resolution_clock::now();
      double elapsed_time_sec =
          std::chrono::duration<float>(t_end - t_start).count();
      std::cout << "At depth " << std::setw(4) << depth << " found score "
                << std::setw(6) << -sr.first << " from a position at "
                << std::setw(4) << depth - sr.second << " plies from root"
                << std::endl;
      std::cout << "         "
                << " nodes:" << std::setw(12) << stats.nodes
                << " bf: " << std::setw(12)
                << std::exp(std::log((double)stats.nodes) / std::max(1, depth))
                << std::endl
                << "         "
                << " gets: " << std::setw(12) << stats.gets
                << " bf: " << std::setw(12)
                << std::exp(std::log((double)stats.gets) / std::max(1, depth))
                << std::endl
                << "         "
                << " hits: " << std::setw(12) << stats.hits
                << " bf: " << std::setw(12)
                << std::exp(std::log((double)stats.hits) / std::max(1, depth))
                << std::endl;
      std::cout << "         "
                << "  time:" << std::setw(12) << elapsed_time_sec << std::endl;
    };

    if (allmoves) {
      Board board(fen, chess960);
      std::vector<std::pair<std::string, int>> result =
          cdbdirect_get(handle, board.getXfen(false));
      for (auto &pair : result) {
        if (pair.first != "a0a0") {
          Move m = cdbuci_to_move(board, pair.first);
          board.makeMove<true>(m);
          std::cout << std::endl
                    << "Analyzing " << uci::moveToUci(m, chess960) << std::endl;
          anaboard(board);
          board.unmakeMove(m);
        }
      }

    } else {
      while (true && depth < 100) {
        anaboard(board);
        depth++;
      }
    }
  }

  std::cout << "Closing DB" << std::endl;
  handle = cdbdirect_finalize(handle);

  return 0;
}
