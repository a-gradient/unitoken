import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host")
    ?? requestHeaders.get("host")
    ?? "localhost:3000";
  const protocol = requestHeaders.get("x-forwarded-proto")
    ?? (host.startsWith("localhost") ? "http" : "https");
  const image_url = new URL("/og.png", `${protocol}://${host}`).href;

  return {
    title: "FFBPE Inspect — Pretokenizer and BPE explorer",
    description:
      "An interactive view of how FFBPE turns text into pretokens, then BPE tokens.",
    openGraph: {
      title: "FFBPE Inspect",
      description: "Explore pretokenizer boundaries and BPE tokens in your browser.",
      images: [{ url: image_url, width: 1536, height: 1024 }],
    },
    twitter: {
      card: "summary_large_image",
      title: "FFBPE Inspect",
      description: "Explore pretokenizer boundaries and BPE tokens in your browser.",
      images: [image_url],
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
