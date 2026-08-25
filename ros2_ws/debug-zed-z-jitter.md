# Debug Session: ZED Z-axis Jitter

- **Session ID**: zed-z-jitter
- **Status**: [OPEN]
- **Symptom**: ID2相对ID0的Z轴波动大，移动相机时Z大幅变化
- **Date**: 2026-07-02

## Current Data Flow
1. PnP解算 → tv[2] (PnP_Z)
2. ZED角点深度采样 → d_med (ZED_Z)
3. 混合: tv[2] = 0.3*PnP_Z + 0.7*ZED_Z
4. 相对位姿: t_rel = R_id0^T * (t_id2 - t_id0)
5. Z轴EMA: 0.7

## Hypotheses
| ID | Hypothesis | Status |
|----|-----------|--------|
| H1 | PnP主因 (表观大小推断Z, 角点敏感) | PENDING |
| H2 | ZED深度主因 (PERFORMANCE模式, 角点跳变) | PENDING |
| H3 | 混合不一致 (两原理各自波动) | PENDING |
| H4 | 相对位姿放大 (两tag Z噪声不相关相减) | PENDING |
| H5 | 相机运动未抵消 (静止时相对Z仍波动) | PENDING |

## Instrumentation Plan
输出每帧: PnP_Z_id0, ZED_Z_id0, blend_Z_id0, PnP_Z_id2, ZED_Z_id2, blend_Z_id2, rel_Z_raw, rel_Z_filtered

## Evidence
(pending collection)
