import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter

def analyze_npz_file(input_file):
    """
    NPZ 파일의 모든 내용을 분석하여 구조와 데이터를 확인
    
    Args:
        input_file (str): 입력 NPZ 파일 경로
    
    Returns:
        dict: 분석 결과   
    """
    
    if not os.path.exists(input_file):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_file}")
        return None
    
    try:
        print(f"📂 파일 분석: {input_file}")
        data = np.load(input_file)
        
        analysis = {}
        
        print(f"📋 사용 가능한 키들: {list(data.files)}")
        print("-" * 50)
        
        for key in data.files:
            arr = data[key]
            analysis[key] = {
                'shape': arr.shape,
                'dtype': arr.dtype,
                'size_mb': arr.nbytes / 1024 / 1024
            }
            
            print(f"🔑 키: '{key}'")
            print(f"   Shape: {arr.shape}")
            print(f"   Data Type: {arr.dtype}")
            print(f"   크기: {arr.nbytes:,} bytes ({arr.nbytes/1024/1024:.2f} MB)")
            
            # 특별한 키들에 대한 추가 분석
            if key == 'ue_pos':
                print(f"   📍 UE Position 정보:")
                if len(arr.shape) == 2:
                    print(f"      총 {arr.shape[0]}개 위치")
                    print(f"      좌표 차원: {arr.shape[1]}D")
                    print(f"      첫 5개 위치:")
                    for i in range(min(5, arr.shape[0])):
                        print(f"        Position {i}: {arr[i]}")
                    print(f"      마지막 5개 위치:")
                    for i in range(max(0, arr.shape[0]-5), arr.shape[0]):
                        print(f"        Position {i}: {arr[i]}")
                        
                    # 위치 통계
                    print(f"      X 범위: [{arr[:, 0].min():.3f}, {arr[:, 0].max():.3f}]")
                    print(f"      Y 범위: [{arr[:, 1].min():.3f}, {arr[:, 1].max():.3f}]")
                    if arr.shape[1] > 2:
                        print(f"      Z 범위: [{arr[:, 2].min():.3f}, {arr[:, 2].max():.3f}]")
                
            elif key == 'cir':
                print(f"   📡 CIR 데이터 정보:")
                print(f"      샘플 수: {arr.shape[0]}")
                if len(arr.shape) > 1:
                    print(f"      채널/안테나: {arr.shape[1]}")
                if len(arr.shape) > 2:
                    print(f"      시간/주파수 차원: {arr.shape[2:]}")
                
                # 복소수 데이터인지 확인
                if np.iscomplexobj(arr):
                    print(f"      ⚠️  복소수 데이터입니다")
                    print(f"      실부 범위: [{np.real(arr).min():.6f}, {np.real(arr).max():.6f}]")
                    print(f"      허부 범위: [{np.imag(arr).min():.6f}, {np.imag(arr).max():.6f}]")
                else:
                    print(f"      데이터 범위: [{arr.min():.6f}, {arr.max():.6f}]")
            
            print()
        
        data.close()
        return analysis
        
    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {e}")
        return None

def compare_positions(scene0_file, scene1_file):
    """
    두 scene 파일의 ue_position을 비교하여 매칭 여부 확인
    Scene1이 더 클 경우 앞부분과 뒷부분을 별도로 분석
    
    Args:
        scene0_file (str): Scene 0 파일 경로
        scene1_file (str): Scene 1 파일 경로
    """
    
    print("🔍 Scene 0과 Scene 1의 위치 정보 상세 비교")
    print("=" * 60)
    
    try:
        # Scene 0 로드
        print("📂 Scene 0 로드...")
        data0 = np.load(scene0_file)
        
        # Scene 1 로드
        print("📂 Scene 1 로드...")
        data1 = np.load(scene1_file)
        
        # ue_pos 확인
        if 'ue_pos' not in data0.files:
            print("❌ Scene 0에 'ue_pos' 키가 없습니다.")
            return
        
        if 'ue_pos' not in data1.files:
            print("❌ Scene 1에 'ue_pos' 키가 없습니다.")
            return
        
        pos0 = data0['ue_pos']
        pos1 = data1['ue_pos']
        cir0 = data0['cir'] if 'cir' in data0.files else None
        cir1 = data1['cir'] if 'cir' in data1.files else None
        
        print(f"📍 Scene 0 위치: {pos0.shape}")
        print(f"📍 Scene 1 위치: {pos1.shape}")
        if cir0 is not None:
            print(f"📡 Scene 0 CIR: {cir0.shape}")
        if cir1 is not None:
            print(f"📡 Scene 1 CIR: {cir1.shape}")
        
        # Scene 0 크기 확인
        scene0_size = pos0.shape[0]  # 289
        scene1_size = pos1.shape[0]  # 578
        
        print(f"\n🔢 데이터 크기 분석:")
        print(f"   Scene 0: {scene0_size}개 샘플")
        print(f"   Scene 1: {scene1_size}개 샘플")
        print(f"   비율: {scene1_size / scene0_size:.2f}배")
        
        if scene1_size >= scene0_size:
            # Scene 1을 두 부분으로 나누어 분석
            pos1_part1 = pos1[:scene0_size]  # 앞 289개
            pos1_part2 = pos1[scene0_size:]  # 뒤 289개
            
            print(f"\n🎯 Scene 1을 두 부분으로 나누어 분석:")
            print(f"   Part 1 (앞 {scene0_size}개): {pos1_part1.shape}")
            print(f"   Part 2 (뒤 {len(pos1_part2)}개): {pos1_part2.shape}")
            
            # Part 1과 Scene 0 비교
            print(f"\n🔍 Scene 0 vs Scene 1 Part 1 비교:")
            match_part1 = np.allclose(pos0, pos1_part1, rtol=1e-5, atol=1e-8)
            print(f"   매칭 결과: {'✅ 일치' if match_part1 else '❌ 불일치'}")
            
            if not match_part1:
                diff = np.abs(pos0 - pos1_part1)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                print(f"   최대 차이: {max_diff:.8f}")
                print(f"   평균 차이: {mean_diff:.8f}")
            
            # Scene 0와 Part 2 비교
            if len(pos1_part2) == scene0_size:
                print(f"\n🔍 Scene 0 vs Scene 1 Part 2 비교:")
                match_part2 = np.allclose(pos0, pos1_part2, rtol=1e-5, atol=1e-8)
                print(f"   매칭 결과: {'✅ 일치' if match_part2 else '❌ 불일치'}")
                
                if not match_part2:
                    diff = np.abs(pos0 - pos1_part2)
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)
                    print(f"   최대 차이: {max_diff:.8f}")
                    print(f"   평균 차이: {mean_diff:.8f}")
            
            # Part 1과 Part 2 비교
            if len(pos1_part2) == scene0_size:
                print(f"\n🔍 Scene 1 Part 1 vs Part 2 비교:")
                match_parts = np.allclose(pos1_part1, pos1_part2, rtol=1e-5, atol=1e-8)
                print(f"   매칭 결과: {'✅ 일치' if match_parts else '❌ 불일치'}")
                
                if not match_parts:
                    diff = np.abs(pos1_part1 - pos1_part2)
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)
                    print(f"   최대 차이: {max_diff:.8f}")
                    print(f"   평균 차이: {mean_diff:.8f}")
            
            # 상세 위치 비교 (처음 10개)
            print(f"\n📋 상세 위치 비교 (처음 10개):")
            for i in range(min(10, scene0_size)):
                print(f"   위치 {i:2d}:")
                print(f"      Scene 0:     {pos0[30 *i]}")
                print(f"      Scene 1 P1:  {pos1_part1[30*i]}")
                if i < len(pos1_part2):
                    print(f"      Scene 1 P2:  {pos1_part2[30*i]}")
                
                # 차이 계산
                diff1 = np.linalg.norm(pos0[30*i] - pos1_part1[30*i])
                diff2 = np.linalg.norm(cir0[30*i] - cir1[30*i])
                print(f"      차이 (0 vs P1): {diff1:.2e}")
                print(f"   cir차이 (0 vs P1): {diff2:.2e}")
                
                if i < len(pos1_part2):
                    diff2 = np.linalg.norm(pos0[30*i] - pos1_part2[30*i])
                    diff_parts = np.linalg.norm(pos1_part1[30*i] - pos1_part2[30*   i])
                    print(f"      차이 (0 vs P2): {diff2:.2e}")
                    print(f"      차이 (P1 vs P2): {diff_parts:.2e}")
                print()
            
            # Part 2의 특성 분석
            if len(pos1_part2) > 0:
                print(f"\n📊 Scene 1 Part 2 특성 분석:")
                print(f"   X 범위: [{pos1_part2[:, 0].min():.3f}, {pos1_part2[:, 0].max():.3f}]")
                print(f"   Y 범위: [{pos1_part2[:, 1].min():.3f}, {pos1_part2[:, 1].max():.3f}]")
                if pos1_part2.shape[1] > 2:
                    print(f"   Z 범위: [{pos1_part2[:, 2].min():.3f}, {pos1_part2[:, 2].max():.3f}]")
                
                # 고유한 위치 개수 확인
                unique_pos_part2 = np.unique(pos1_part2, axis=0)
                print(f"   고유한 위치 개수: {len(unique_pos_part2)}")
                
                # Part 2가 반복인지 확인
                if len(pos1_part2) == scene0_size:
                    # Scene 0와 비교
                    unique_pos_scene0 = np.unique(pos0, axis=0)
                    print(f"   Scene 0 고유 위치: {len(unique_pos_scene0)}")
                    
                    # 동일한 고유 위치들인지 확인
                    if len(unique_pos_part2) == len(unique_pos_scene0):
                        # 각 고유 위치가 일치하는지 확인
                        distances = []
                        for up2 in unique_pos_part2:
                            min_dist = min([np.linalg.norm(up2 - us0) for us0 in unique_pos_scene0])
                            distances.append(min_dist)
                        max_dist = max(distances) if distances else float('inf')
                        print(f"   고유 위치 간 최대 거리: {max_dist:.2e}")
                        
                        if max_dist < 1e-6:
                            print("   ✅ Part 2는 Scene 0와 같은 위치들의 재배열/반복으로 보입니다!")
        
        # 시각화 (2D인 경우)
        if pos0.shape[1] >= 2 and pos1.shape[1] >= 2:
            visualize_positions_detailed(pos0, pos1, scene0_size)
        
        data0.close()
        data1.close()
        
    except Exception as e:
        print(f"❌ 비교 중 오류 발생: {e}")

def visualize_positions_detailed(pos0, pos1, scene0_size):
    """
    Scene 0, Scene 1 Part 1, Scene 1 Part 2를 상세히 시각화
    """
    try:
        pos1_part1 = pos1[:scene0_size]
        pos1_part2 = pos1[scene0_size:] if len(pos1) > scene0_size else None
        
        if pos1_part2 is not None and len(pos1_part2) > 0:
            # 3개 subplot
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # Scene 0
            axes[0].scatter(pos0[:, 0], pos0[:, 1], c='blue', alpha=0.7, s=50)
            axes[0].set_title(f'Scene 0 ({len(pos0)} samples)')
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')
            axes[0].grid(True, alpha=0.3)
            
            # 위치 번호 표시 (처음 20개만)
            for i in range(min(20, len(pos0))):
                axes[0].annotate(str(i), (pos0[i, 0], pos0[i, 1]), 
                            xytext=(5, 5), textcoords='offset points', 
                            fontsize=8, alpha=0.7)
            
            # Scene 1 Part 1
            axes[1].scatter(pos1_part1[:, 0], pos1_part1[:, 1], c='red', alpha=0.7, s=50)
            axes[1].set_title(f'Scene 1 Part 1 ({len(pos1_part1)} samples)')
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Y')
            axes[1].grid(True, alpha=0.3)
            
            for i in range(min(20, len(pos1_part1))):
                axes[1].annotate(str(i), (pos1_part1[i, 0], pos1_part1[i, 1]), 
                            xytext=(5, 5), textcoords='offset points', 
                            fontsize=8, alpha=0.7)
            
            # Scene 1 Part 2
            axes[2].scatter(pos1_part2[:, 0], pos1_part2[:, 1], c='green', alpha=0.7, s=50)
            axes[2].set_title(f'Scene 1 Part 2 ({len(pos1_part2)} samples)')
            axes[2].set_xlabel('X')
            axes[2].set_ylabel('Y')
            axes[2].grid(True, alpha=0.3)
            
            for i in range(min(20, len(pos1_part2))):
                axes[2].annotate(str(i), (pos1_part2[i, 0], pos1_part2[i, 1]), 
                            xytext=(5, 5), textcoords='offset points', 
                            fontsize=8, alpha=0.7)
        else:
            # 2개 subplot (기존 코드)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Scene 0
            axes[0].scatter(pos0[:, 0], pos0[:, 1], c='blue', alpha=0.7, s=50)
            axes[0].set_title(f'Scene 0 ({len(pos0)} samples)')
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')
            axes[0].grid(True, alpha=0.3)
            
            for i in range(min(20, len(pos0))):
                axes[0].annotate(str(i), (pos0[i, 0], pos0[i, 1]), 
                            xytext=(5, 5), textcoords='offset points', 
                            fontsize=8, alpha=0.7)
            
            # Scene 1 Part 1
            axes[1].scatter(pos1_part1[:, 0], pos1_part1[:, 1], c='red', alpha=0.7, s=50)
            axes[1].set_title(f'Scene 1 Part 1 ({len(pos1_part1)} samples)')
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Y')
            axes[1].grid(True, alpha=0.3)
            
            for i in range(min(20, len(pos1_part1))):
                axes[1].annotate(str(i), (pos1_part1[i, 0], pos1_part1[i, 1]), 
                            xytext=(5, 5), textcoords='offset points', 
                            fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('data/position_detailed_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("📊 상세 위치 시각화가 'position_detailed_comparison.png'에 저장되었습니다.")
        
        # 추가: 오버레이 비교 플롯
        plt.figure(figsize=(10, 8))
        plt.scatter(pos0[:, 0], pos0[:, 1], c='blue', alpha=0.6, s=60, label='Scene 0', marker='o')
        plt.scatter(pos1_part1[:, 0], pos1_part1[:, 1], c='red', alpha=0.6, s=40, label='Scene 1 Part 1', marker='^')
        
        if pos1_part2 is not None and len(pos1_part2) > 0:
            plt.scatter(pos1_part2[:, 0], pos1_part2[:, 1], c='green', alpha=0.6, s=40, label='Scene 1 Part 2', marker='s')
        
        plt.title('All Positions Overlay Comparison')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('data/position_overlay_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("📊 오버레이 위치 비교가 'position_overlay_comparison.png'에 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 상세 시각화 중 오류 발생: {e}")

def visualize_positions(pos0, pos1):
    """
    두 scene의 위치를 시각화 (기존 함수는 호환성 유지)
    """
    scene0_size = len(pos0)
    visualize_positions_detailed(pos0, pos1, scene0_size)

def extract_cir_data(input_file, output_file):
    """
    NPZ 파일에서 CIR 데이터만 추출하여 NPY 파일로 저장
    
    Args:
        input_file (str): 입력 NPZ 파일 경로
        output_file (str): 출력 NPY 파일 경로
    """
    
    # 입력 파일 존재 확인
    if not os.path.exists(input_file):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_file}")
        return False
    
    try:
        # NPZ 파일 로드
        print(f"📂 입력 파일 로드: {input_file}")
        data = np.load(input_file)
        
        # CIR 데이터 추출
        if 'cir' not in data.files:
            print("❌ 'cir' 키를 찾을 수 없습니다.")
            print(f"   사용 가능한 키: {list(data.files)}")
            data.close()
            return False
        
        cir_data = data['cir']
        print(f"📊 CIR 데이터 정보:")
        print(f"   Shape: {cir_data.shape}")
        print(f"   Data Type: {cir_data.dtype}")
        print(f"   메모리 사용량: {cir_data.nbytes:,} bytes ({cir_data.nbytes/1024/1024:.2f} MB)")
        
        # 데이터 닫기
        data.close()
        
        # 출력 디렉토리 생성 (필요한 경우)
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📁 출력 디렉토리 생성: {output_dir}")
        
        # NPY 파일로 저장
        print(f"💾 CIR 데이터 저장: {output_file}")
        np.save(output_file, cir_data)
        
        # 저장된 파일 확인
        saved_size = os.path.getsize(output_file)
        print(f"✅ 저장 완료!")
        print(f"   출력 파일 크기: {saved_size:,} bytes ({saved_size/1024/1024:.2f} MB)")
        
        # 저장된 파일 검증
        print(f"🔍 저장된 파일 검증...")
        loaded_cir = np.load(output_file)
        print(f"   검증 Shape: {loaded_cir.shape}")
        print(f"   검증 Data Type: {loaded_cir.dtype}")
        print(f"   데이터 일치 여부: {np.array_equal(cir_data, loaded_cir)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

def extract_with_position_info(input_file, output_cir_file, output_pos_file=None, output_obj_file=None):
    """
    NPZ 파일에서 CIR 데이터와 위치 정보를 함께 추출
    
    Args:
        input_file (str): 입력 NPZ 파일 경로
        output_cir_file (str): CIR 데이터 출력 파일
        output_pos_file (str): 위치 데이터 출력 파일 (None이면 자동 생성)
        output_obj_file (str): 객체 데이터 출력 파일 (None이면 자동 생성)
    """
    
    if output_pos_file is None:
        output_pos_file = output_cir_file.replace('_cir.npy', '_positions.npy')
    if output_obj_file is None:
        output_obj_file = output_cir_file.replace('_cir.npy', '_objects.npy')
    if not os.path.exists(input_file):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_file}")
        return False
    
    try:
        data = np.load(input_file)
        
        # CIR 데이터 추출
        if 'cir' in data.files:
            cir_data = data['cir']
            np.save(output_cir_file, cir_data)
            print(f"✅ CIR 데이터 저장: {output_cir_file}")
        
        # 위치 데이터 추출
        if 'ue_pos' in data.files:
            pos_data = data['ue_pos']
            np.save(output_pos_file, pos_data)
            print(f"✅ 위치 데이터 저장: {output_pos_file}")
        
        # 객체 데이터 추출
        if 'obj_pos' in data.files:
            obj_data = data['obj_pos']
            np.save(output_obj_file, obj_data)
            print(f"✅ 객체 데이터 저장: {output_obj_file}")
        
        data.close()
        return True
        
    except Exception as e:
        print(f"❌ 추출 중 오류 발생: {e}")
        return False

def analyze_scene1_structure(scene1_file):
    """
    Scene 1의 내부 구조를 상세히 분석
    578개 샘플이 어떻게 구성되어 있는지 확인
    """
    print("🔬 Scene 1 내부 구조 상세 분석")
    print("=" * 60)
    
    try:
        data = np.load(scene1_file)
        
        if 'ue_pos' not in data.files or 'cir' not in data.files:
            print("❌ 필요한 데이터가 없습니다.")
            return
        
        pos = data['ue_pos']
        cir = data['cir']
        
        print(f"📊 기본 정보:")
        print(f"   위치 데이터: {pos.shape}")
        print(f"   CIR 데이터: {cir.shape}")
        
        # 고유한 위치들 찾기
        unique_positions = np.unique(pos, axis=0)
        print(f"   고유한 위치 개수: {len(unique_positions)}")
        
        # 각 위치별 샘플 개수 계산
        position_counts = {}
        position_indices = {}
        
        for i, position in enumerate(pos):
            pos_key = tuple(position)
            if pos_key not in position_counts:
                position_counts[pos_key] = 0
                position_indices[pos_key] = []
            position_counts[pos_key] += 1
            position_indices[pos_key].append(i)
        
        print(f"\n📋 각 위치별 샘플 분포:")
        counts_list = list(position_counts.values())
        print(f"   최소 샘플 수: {min(counts_list)}")
        print(f"   최대 샘플 수: {max(counts_list)}")
        print(f"   평균 샘플 수: {np.mean(counts_list):.2f}")
        print(f"   위치당 샘플 수 분포: {sorted(set(counts_list))}")
        
        # 샘플 수별 위치 개수
        count_distribution = Counter(counts_list)
        print(f"\n📈 샘플 수별 위치 개수:")
        for sample_count, num_positions in sorted(count_distribution.items()):
            print(f"   {sample_count}개 샘플을 가진 위치: {num_positions}개")
        
        # 처음 몇 개 위치의 상세 정보
        print(f"\n🔍 처음 10개 고유 위치의 상세 정보:")
        for i, (pos_key, indices) in enumerate(list(position_indices.items())[:10]):
            print(f"   위치 {i}: {pos_key}")
            print(f"      샘플 개수: {len(indices)}")
            print(f"      인덱스: {indices[:5]}{'...' if len(indices) > 5 else ''}")
        
        # 289개씩 나누어서 분석
        part1_indices = list(range(289))
        part2_indices = list(range(289, 578))
        
        pos_part1 = pos[part1_indices]
        pos_part2 = pos[part2_indices]
        
        print(f"\n🎯 289개씩 나누어 분석:")
        print(f"   Part 1 (0-288): 고유 위치 {len(np.unique(pos_part1, axis=0))}개")
        print(f"   Part 2 (289-577): 고유 위치 {len(np.unique(pos_part2, axis=0))}개")
        
        # Part 1과 Part 2의 위치 분포 비교
        unique_part1 = np.unique(pos_part1, axis=0)
        unique_part2 = np.unique(pos_part2, axis=0)
        
        # 공통 위치 찾기
        common_positions = []
        for up1 in unique_part1:
            for up2 in unique_part2:
                if np.allclose(up1, up2, rtol=1e-8, atol=1e-10):
                    common_positions.append(up1)
                    break
        
        print(f"   Part 1과 Part 2의 공통 위치: {len(common_positions)}개")
        
        # 순서 분석 - Part 1과 Part 2가 같은 순서인지 확인
        if len(pos_part1) == len(pos_part2):
            sequence_match = np.allclose(pos_part1, pos_part2, rtol=1e-8, atol=1e-10)
            print(f"   Part 1과 Part 2의 순서 일치: {'✅ 예' if sequence_match else '❌ 아니오'}")
            
            if not sequence_match:
                # 몇 개나 다른지 확인
                diff_count = 0
                for i in range(len(pos_part1)):
                    if not np.allclose(pos_part1[i], pos_part2[i], rtol=1e-8, atol=1e-10):
                        diff_count += 1
                print(f"   순서가 다른 위치 개수: {diff_count}/{len(pos_part1)}")
        
        # 연속된 샘플들의 패턴 분석
        print(f"\n🔄 연속 샘플 패턴 분석:")
        consecutive_same = 0
        max_consecutive = 0
        current_consecutive = 1
        
        for i in range(1, len(pos)):
            if np.allclose(pos[i], pos[i-1], rtol=1e-10, atol=1e-12):
                current_consecutive += 1
                consecutive_same += 1
            else:
                max_consecutive = max(max_consecutive, current_consecutive)
                current_consecutive = 1
        
        max_consecutive = max(max_consecutive, current_consecutive)
        
        print(f"   연속으로 같은 위치인 샘플 쌍: {consecutive_same}개")
        print(f"   최대 연속 같은 위치 길이: {max_consecutive}")
        
        # Reference Point 패턴 추정
        print(f"\n📍 Reference Point 패턴 추정:")
        
        # 각 위치가 몇 번씩 나타나는지 확인
        if len(set(counts_list)) == 1:
            samples_per_rp = counts_list[0]
            estimated_rps = len(unique_positions)
            print(f"   추정: {estimated_rps}개 RP, 각각 {samples_per_rp}개 샘플")
            print(f"   총 샘플: {estimated_rps * samples_per_rp} (실제: {len(pos)})")
        else:
            print(f"   불규칙한 샘플 분포로 보입니다.")
            print(f"   가능한 해석들:")
            if 2 in set(counts_list):
                num_double = sum(1 for c in counts_list if c == 2)
                num_single = sum(1 for c in counts_list if c == 1)
                print(f"     - {num_double}개 RP는 2번씩, {num_single}개 RP는 1번씩 샘플링")
        
        data.close()
        
        return {
            'total_samples': len(pos),
            'unique_positions': len(unique_positions),
            'position_counts': position_counts,
            'samples_per_position': counts_list
        }
        
    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {e}")
        return None

# main 함수
if __name__ == "__main__":
    
    print("🚀 NPZ 파일 상세 분석 및 CIR 데이터 추출")
    print("=" * 70)
    
    # 1. Scene 0 분석
    scene0_file = "data/IndoorChannelGenerator_scene0.npz"
    scene1_file = "data/IndoorChannelGenerator_scene1.npz"
    
    print("\n📊 Scene 0 상세 분석")
    print("-" * 50)
    analysis0 = analyze_npz_file(scene0_file)
    
    print("\n📊 Scene 1 상세 분석")
    print("-" * 50)
    analysis1 = analyze_npz_file(scene1_file)
    
    # 2. Scene 1 내부 구조 분석 (새로 추가)
    print("\n" + "=" * 70)
    scene1_structure = analyze_scene1_structure(scene1_file)
    
    # 3. 위치 정보 비교
    print("\n" + "=" * 70)
    compare_positions(scene0_file, scene1_file)
    
    # 4. CIR 데이터와 위치 정보 추출
    print("\n" + "=" * 70)
    print("💾 데이터 추출")
    print("-" * 50)
    
    # Scene 0 추출
    extract_with_position_info(scene0_file, "data/scene0_cir.npy", "data/scene0_positions.npy", "data/scene0_objects.npy")
    
    # Scene 1 추출
    extract_with_position_info(scene1_file, "data/scene1_cir.npy", "data/scene1_positions.npy", "data/scene1_objects.npy")
    
    print("\n🎉 모든 작업이 완료되었습니다!")
    print("📋 생성된 파일들:")
    print("   - scene0_cir.npy: Scene 0 CIR 데이터")
    print("   - scene0_positions.npy: Scene 0 위치 정보")
    print("   - scene1_cir.npy: Scene 1 CIR 데이터") 
    print("   - scene1_positions.npy: Scene 1 위치 정보")
    print("   - position_detailed_comparison.png: 상세 위치 비교")
    print("   - position_overlay_comparison.png: 오버레이 위치 비교")
