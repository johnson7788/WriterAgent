

import { env } from "~/env";

export function resolveServiceURL(path: string) {
  let BASE_URL = env.NEXT_PUBLIC_API_URL ?? "http://localhost:7800";
  console.log("当前环境变量 NEXT_PUBLIC_API_URL:", env.NEXT_PUBLIC_API_URL);
  if (!BASE_URL.endsWith("/")) {
    BASE_URL += "/";
  }
  console.log("请求后端服务 URL:", BASE_URL);
  return new URL(path, BASE_URL).toString();
}
