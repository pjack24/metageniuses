import { useState, useEffect } from "react";

const BASE = import.meta.env.BASE_URL || "/";

export default function useApi(url) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Map /api/experiments/N → static JSON files
  const staticUrl = url.replace(/^\/api\/experiments\/(\d+)$/, `${BASE}data/experiment$1.json`);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetch(staticUrl)
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        return res.json();
      })
      .then((json) => {
        if (!cancelled) {
          setData(json);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      });

    return () => { cancelled = true; };
  }, [staticUrl]);

  return { data, loading, error };
}
