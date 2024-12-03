import { spawn } from "child_process";
import os from "os";

const platform = os.platform(); // 현재 운영 체제 확인

const openTerminal = (cmd, cwd) => {
  if (platform === "win32") {
    // Windows용
    spawn("cmd.exe", ["/c", `start cmd.exe /k "cd ${cwd} && ${cmd}"`], { shell: true });
  } else if (platform === "darwin") {
    // macOS용
    spawn("osascript", [
      "-e",
      `tell application "Terminal" to do script "cd ${cwd} && ${cmd}"`
    ]);
  } else {
    console.error("Unsupported OS. This script supports only Windows and macOS.");
  }
};

// 백엔드 실행
openTerminal("python server.py", "backend");

// 프론트엔드 실행
openTerminal("npm run dev", "frontend");
