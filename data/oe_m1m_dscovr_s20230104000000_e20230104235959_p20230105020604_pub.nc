CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230104000000_e20230104235959_p20230105020604_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-01-05T02:06:04.085Z   date_calibration_data_updated         2022-11-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-01-04T00:00:00.000Z   time_coverage_end         2023-01-04T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill         �   records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxW�=�  T          AAp�@��A�H�e��Q�Bm�@��A(z�J=q�o\)Bs�R                                    BxW�Lf  �          AC33@�Q�A
=�l����\)BkG�@�Q�A)��c�
��ffBqff                                    BxW�[  �          AG�
@��HA#��q�����Bl��@��HA-녿c�
���\Bs                                      BxW�i�  �          AJ{@�=qA#�
�tz����RBiG�@�=qA.=q�n{���RBoz�                                    BxW�xX  T          AI@�33A&{�qG����Bn=q@�33A0(��Y���vffBt{                                    BxW���  �          AI��@�=qA%��p������Bn��@�=qA0  �W
=�u�Btff                                    BxW���  �          AJ�\@�\)A$Q���  ��\)Bk{@�\)A/��������Bq��                                    BxW��J              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���  �          AI@�  A!�������
Bh@�  A-���H��z�BpQ�                                    BxW��<  T          AHz�@���A�R��=q���Bf��@���A+\)��G���Bnz�                                    BxW���  �          AJ=q@��A"{�mp����HBdQ�@��A,  �^�R�~{Bjz�                                    BxW��  T          AJ�H@�
=A"=q�c�
��Q�Ba�H@�
=A+��8Q��P��Bg�                                    BxW��.  T          AF�H@�Q�A   �^{���
Bc�
@�Q�A)��.{�HQ�Bi�                                    BxW�
�  T          AJ=q@�{A ���g����Baz�@�{A*�\�O\)�k�Bg�\                                   BxW�z  �          AL��@��HA$  �w�����Be{@��HA.�\��  ��
=Bkz�                                   BxW�(   �          AI��@�z�A#\)�l(���z�Bg�R@�z�A-��Y���vffBm�                                    BxW�6�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�El              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�T            A<(�@��
Aff��
=����Be=q@��
A   ����33Bnff                                    BxW�b�  �          A?�@��A��p���Q�Bb@��A!p����3�Bm=q                                    BxW�q^  �          A@��@��A�������HBc�H@��A"�R�\)�,(�Bn
=                                    BxW��  
�          A>ff@ÅAff��Q���{Ba
=@ÅA Q�����Bjp�                                    BxW���  
�          A<  @���A�������RB`�@���A{�z��"=qBjp�                                    BxW��P  "          A4��@�{A\)���H��B^��@�{A�����Bg�R                                    BxW���  S          A2�R@��A{�QG���(�Baff@��A�H�L����(�Bg��                                    BxW���  �          A2�\@��A���Q���
=B_�@��A녿W
=���\Be�\                                    BxW��B  
�          A3�
@�=qA\)�k���
=B\z�@�=qA녿�  ���BdG�                                    BxW���  �          A:=q@���A�H���
����B_�\@���A(���\)��Bh�                                    BxW��  
(          A;
=@��HA(���z���  Bh@��HA���9G�Bs                                      BxW��4  $          A:�\@���A�
��ff��33Bi�
@���A��=q�?33Bt=q                                    BxW��  �          A<z�@��\A{��{��=qBjG�@��\A!�����9G�Btff                                    BxW��  "          A<��@��HA\)�������Bk  @��HA"=q����+�
Bt��                                    BxW�!&  "          A<(�@�Q�A�\��z��ģ�Bk�
@�Q�A!��z��5Bu�                                    BxW�/�  T          A8��@�Q�A=q�����BpQ�@�Q�A �����)��Byz�                                    BxW�>r  
�          A-@�ffA�H��z�����BiG�@�ffA���G��,Q�Br�                                    BxW�M  "          A0��@���@��\��{�ԏ\BQ�R@���A���.{�d(�B_                                      BxW�[�  
�          A.�\@��\@�33���\��=qBU�R@��\Ap��'
=�]p�Bbz�                                    BxW�jd  T          A*ff@�{@�(��������B]��@�{A����S
=Biff                                    BxW�y
  
�          A*�H@��R@����H��p�BT�@��RA	�+��h��Ba��                                    BxW���  �          A+�@��R@�Q��������
BS  @��RA���8���z�\Ba�                                    BxW��V  
*          A,��@�ff@����  �хBW�\@�ffAz��#�
�[�
Bd{                                    BxW���  
X          A-�@�Ap���G��ƣ�Bh�@�A��{�=G�Br�H                                    BxW���  �          A,��@�G�A	���Q���\)BsG�@�G�A33�Q��5B|�\                                    BxW��H  "          A-@���A	G����
��33Bs33@���A�
��R�=B|�R                                    BxW���  
�          A/33@��HA
�H���H��z�Bsz�@��HAG��(��8Q�B|                                    BxW�ߔ  "          A0  @���A(����H�ř�Bu  @���Aff�
�H�6=qB~�                                    BxW��:  �          A0��@��HAp���Q����RBu33@��HA\)�z��,��B}��                                    BxW���  
�          A/
=@��A��������Bu�@��A{��p��'33B}�R                                    BxW��  �          A.ff@�  A{��p����B|�\@�  A���p��'�B�W
                                    BxW�,  V          A,z�@��A�����\��B(�@��A�R��z��#33B�z�                                    BxW�(�             A-G�@���A
�H��Q�����Bw\)@���A���Q��5�B��                                    BxW�7x  �          A,��@�p�A
�R��p�����B{�H@�p�AG����C\)B�k�                                    BxW�F  �          A,  @��A	G���\)����B{=q@��A(����K�
B�=q                                    BxW�T�  �          A,z�@�A���Q���=qB|=q@�AG��Q��6ffB�aH                                    BxW�cj  �          A+33@��RA��������HB|�@��RA���(��  B�\                                    BxW�r  T          A,Q�@���Aff�������
B|�@���A=q��\)�
�\B���                                    BxW���  T          A,��@��A��\)���B}��@��A33�����p�B�L�                                    BxW��\  �          A+33@�z�A����33��z�B~z�@�z�A�����H�33B��H                                    BxW��  �          A+
=@�G�A  ��(�����BzQ�@�G�A(���G���B���                                    BxW���  T          A)��@�{A
�R���R��
=B{z�@�{A33��\)�"{B��                                    BxW��N  �          A(��@�z�A	�������B|�@�z�A�\��z��'33B�
=                                    BxW���  �          A'\)@��A��������
=B{�@��AG����"�\B��H                                    BxW�ؚ  �          A&�R@�Q�A��������
Bx�@�Q�A��޸R�=qB�                                    BxW��@  �          A*{@��
A
�\���\���\Bw�R@��
A�\��G��(�Bp�                                    BxW���  �          A)�@��A�������=qBk�H@��A���5�Bu33                                    BxW��  T          A(Q�@�z�@��R���R���HB\(�@�z�A
ff�'��f=qBhG�                                    BxW�2  �          A'�
@�ff@�ff���
��p�B_�
@�ffA
�H�1��u��BlG�                                    BxW�!�  �          A'\)@��@�(���z��ޣ�B^(�@��A	�3�
�y�Bj��                                    BxW�0~  �          A%@��@�\)������p�BYp�@��A�H�/\)�up�BfG�                                    BxW�?$  �          A%G�@�33@����=q��z�BX��@�33A{�3�
�|Q�Be                                    BxW�M�  �          A&�R@��@�Q����
���BZ��@��A�
�5��|z�Bg�                                    BxW�\p  �          A"=q@���@�p����
����B\�
@���Ap��'��o\)Bi�                                    BxW�k  �          A#
=@�@�{������\)B\@�A��)���p��Bi�                                    BxW�y�  �          A!�@��R@�G����H����BY�R@��RA\)�(Q��q�BfG�                                    BxW��b  �          A!��@�
=@�33��=q����BZz�@�
=A(��%�m�Bf��                                    BxW��  �          A"�H@��@�p���\)�ݙ�B]�H@��A��/\)�yBj\)                                    BxW���  �          A#\)@�  @�{��(���\B`�@�  A�R�8Q���33BmG�                                    BxW��T  �          A#
=@��H@�z����H����B]�
@��HA�7
=��=qBj�R                                    BxW���  �          A"=q@���@�z��������\BS��@���Ap��7
=���HBa=q                                    BxW�Ѡ  �          A ��@��@���  ��{BNz�@��@�z��8����{B\�                                    BxW��F  
�          A�H@��H@�=q��=q��33BJ�\@��H@�\)�0  ���\BX�                                    BxW���  
�          A\)@�  @�\)��(����BJ�
@�  @�
=�E���BZ{                                    BxW���  �          A
=@�=q@�{��
=�\)BEQ�@�=q@�G��b�\��33BWp�                                    BxW�8  �          Aff@���@ȣ���
=�{BJG�@���@�33�a���p�B[��                                    BxW��  
�          A33@�(�@���z����BM��@�(�@�  �Z=q��z�B^�                                    BxW�)�  �          A{@��@�ff��\)���BN��@��@�33�n�R����B`ff                                    BxW�8*  �          A$  @�Q�@����  �p�BN�
@�Q�@���|����G�B`�H                                    BxW�F�  �          A%��@�z�@�  ��p���=qBQ��@�z�A�c33��Q�Ba�\                                    BxW�Uv  T          A-�@�A�i�����HBZz�@�A�
��G���
=BbQ�                                    BxW�d  �          A2�H@�(�A{������Q�BWG�@�(�A=q���R�$(�B`��                                    BxW�r�  �          A0Q�@�A���J�H��BXp�@�A�ÿ�  ��\)B^�R                                    BxW��h  �          A2�R@��
A{�(��5G�BYz�@��
Aff<��
>�B\��                                    BxW��  �          A0Q�@ʏ\A녿Ǯ�=qBY�H@ʏ\A�
?!G�@Q�B[G�                                    BxW���  �          A-��@���A{�xQ����B]33@���A��?���@ʏ\B\�H                                    BxW��Z  �          A-p�@�33A�
�fff��Q�BW�@�33A33?�p�@�Q�BWp�                                    BxW��   �          A,(�@�A(��k���p�B[�@�A�?��H@�{BZ�R                                    BxW�ʦ  �          A-��@ȣ�A�ÿY����
=BZ(�@ȣ�A(�?��@ڏ\BYz�                                    BxW��L  �          A.=q@ʏ\Ap��
=q�7�BY��@ʏ\A\)?˅A{BX
=                                    BxW���  �          A.{@��
A�þ����ǮBX��@��
A
{?���ABVQ�                                    BxW���  �          A/\)@�Q�A��k���
=BV33@�Q�A	�?��A=qBS�                                    BxW�>  �          A2�\@�Q�A�
�xQ���  BXG�@�Q�A\)?�Q�@��BW��                                    BxW��  T          A0��@�33Ap�����5�BU{@�33A�?���A�RBS�                                    BxW�"�  �          A/\)@�z�A=q�L�Ϳ��BY(�@�z�A
�H?�z�A!�BV��                                    BxW�10  T          A/33@ϮA
�R?��
A ��BT�
@ϮA�@b�\A�
=BL                                    BxW�?�  �          A.�H@��HA=q@��AI�BOp�@��H@��H@���A�\)BD(�                                    BxW�N|  �          A.ff@�=qA
=q@�A0  BW\)@�=q@��@��A���BM�\                                    BxW�]"  �          A.{@ʏ\Az�?��
@�{BX��@ʏ\A��@Dz�A�
=BR�                                    BxW�k�  �          A.�R@��A
�R�J=q���
Bc�@��A�R�����Q�Bi\)                                    BxW�zn  |          AB�H?���@��H�6ff�HB�8R?���@�z��"{�_��B��H                                    BxW��  �          AJff���?��R�H��¨#�B����@�z��>�R�=B�\)                                    BxW���  �          AD�׿u>�{�C�ª� C W
�u@K��<���\BՀ                                     BxW��`  "          ADz�O\)>�G��B�H«aHCk��O\)@P���;�
{B�Ǯ                                    BxW��  �          AD�;�(����Dz�°  C7\)��(�@5��?33�B�G�                                    BxW�ì  �          AB{����k��AG�©� C?녿��@$z��<��\)B�R                                    BxW��R  �          AB�R��녿�z��@z�¤��Ca�f���?�33�?�¡  B�{                                    BxW���  �          AH�þ�33?�G��G\)¥�)B�녾�33@��H�<z�p�B�#�                                    BxW��  T          AK33��=q?\�I��¦B�B���=q@�(��>�\�\B��                                    BxW��D  T          AK���?����J=q¨(�B�\��@�=q�@z�L�B¨�                                    BxW��  �          AK���R?�=q�J{¦�qB�׿�R@�{�?�
{BĽq                                    BxW��  �          AK
=�+�?�
=�Ip�¥�B�q�+�@����>�H#�BŮ                                    BxW�*6  �          AK��0��?��I�¢u�Bݔ{�0��@�(��=���B��f                                    BxW�8�  �          AK��0��?�\�IG�¢�
B�uÿ0��@��\�=��B�B�\                                    BxW�G�  �          AK33����?�p��I�£p�B�녿���@�G��>ff33B�{                                    BxW�V(  T          AJ�H��(�?����G��)B�� ��(�@�(��;�
��B�k�                                    BxW�d�  h          AHQ����=u�A���
C2�=���@2�\�<z�{C                                    BxW�st  "          AK\)��  �W
=�J�R±
=C[E��  @)���F=q�RB��f                                    BxW��  �          AK\)=��
�W
=�K
=±�C���=��
@(���F�\��B��                                    BxW���  �          AL��@	���n{�HQ���C�P�@	��?�33�F�\��B&{                                    BxW��f  T          AO�
@L(�?(��H����A-�@L(�@Y���Ap���B;G�                                    BxW��  
�          AP��@=p�@.{�E�B*�@=p�@���7
=�v��Bu\)                                    BxW���  "          APz�?�\@�z��C33z�B���?�\@�  �/��f(�B��\                                    BxW��X  
Z          APz�?z�H@�33�EG�8RB�  ?z�H@׮�1�j  B�\)                                    BxW���  �          AP��?��@���E�z�B���?��@�G��1p��h�
B��q                                    BxW��  
�          AQp�?s33@����C�
�3B�8R?s33@��/
=�c
=B�                                    BxW��J  �          AR{?�ff@q��G�
z�B��?�ff@�{�5���o�RB��                                    BxW��  �          AQ?���@l(��H(�33B��q?���@�33�6=q�q\)B��{                                    BxW��  
�          AQ�?z�H@s�
�G\)  B�G�?z�H@�ff�5G��pG�B���                                    BxW�#<  T          AP��>�@{��F�H�3B�>�@љ��4Q��o�B���                                    BxW�1�  
�          AQ�>�(�@{��G33��B���>�(�@љ��4���o\)B��=                                    BxW�@�  �          ANff���@/\)�G�L�B��Ϳ��@�z��9���B���                                    BxW�O.  "          AO
=��\@S�
�G��Bŏ\��\@�{�7\)�z{B��
                                    BxW�]�  "          AO\)���\@��K\)�\B�aH���\@�(��?
=8RB�#�                                    BxW�lz  �          AO33���R@&ff�I�Q�B�
=���R@�Q��<Q��B΅                                    BxW�{   "          AP  ��33@1��JffB�B��f��33@��<(�k�B��
                                    BxW���  "          AO����@\)�K
=��B�(����@����=B��                                    BxW��l  �          AP  ��
=@���K�B�B�8R��
=@�=q�>�\��B�#�                                    BxW��  
�          AP  �E�@L(��IG��RB�=q�E�@����9�|�B�.                                    BxW���  �          APz�E�@<(��J�R
=B�aH�E�@�=q�<(��HB�                                    BxW��^  
�          AP(��k�@0  �J�HG�B��f�k�@��
�<��p�B�\)                                    BxW��  "          APz�aG�@-p��K��qB��)�aG�@��H�=���Bƨ�                                    BxW��  T          APz�xQ�@1��K33�Bڮ�xQ�@�z��=�G�B�z�                                    BxW��P  T          APQ쿃�
@2�\�J�R�{B܏\���
@�z��<��{Bɣ�                                    BxW���  �          APzῠ  @;��J=q�B�W
��  @����;�
�=B͞�                                    BxW��  "          APz�s33@H���I��Bծ�s33@��R�:�R�~�B��                                    BxW�B  �          APz�h��@H���I���B�Q�h��@�ff�;
=�~�
B��                                    BxW�*�  �          AP�Ϳ�Q�@@���J=qL�B�(���Q�@�=q�;�
=qB��                                    BxW�9�  T          APQ쿥�@;��I��{B㙚���@�\)�;�
ǮB΀                                     BxW�H4  
�          AP  ��G�@(Q��J=q
=B��G�@�{�=G��fB�aH                                    BxW�V�  T          AP�׿��@=p��J{#�B�Ǯ���@�  �<  �=B�p�                                    BxW�e�  "          APQ쿹��@:=q�I��{B��f����@�{�;�
B���                                    BxW�t&  
�          APzῗ
=@:�H�J{��B��)��
=@�ff�<(�L�B�aH                                    BxW���  �          AP�ÿ��\@Mp��I�B�aH���\@��R�;
=�~ffB�L�                                    BxW��r  �          AP�׿�@�33�E�B��)��@����4(��o(�B�                                      BxW��  �          AQ���L��@�  �F=q\B�uþL��@��4Q��mQ�B�z�                                    BxW���  T          AR=q�8Q�@vff�H��L�B���8Q�@���8(��tffB�                                      BxW��d  "          AR�H�fff@u�IG�.B�\)�fff@ə��8���t��B�B�                                    BxW��
  T          AR�R���@\)�G��3B�G����@���6�\�p�\B˳3                                    BxW�ڰ  �          AR�H���@��R�F�\��B�=q���@Ӆ�5��lB��H                                    BxW��V  �          AT(��\)@���HQ��qB½q�\)@�
=�6�\�m��B�z�                                    BxW���  "          AT�׿��@����G���B�Ǯ���@�{�4���iz�B�\                                    BxW��  T          ATQ�#�
@�Q��E��\B�W
�#�
@��
�2�H�e��B�G�                                    BxW�H  
Z          AS\)��
=@����D��u�B����
=@��1��effB���                                    BxW�#�  �          AT�Ϳ�R@��H�DQ���B���R@���0Q��_��B���                                    BxW�2�  T          AU녿333@��
�H(�
=B���333@߮�5�i�B�\)                                    BxW�A:  �          AU녿&ff@����D(��B�\�&ff@�\�/�
�]Q�B���                                    BxW�O�  T          AV�R�=p�@�Q��E�aHB�
=�=p�@���0���^{B�33                                    BxW�^�  �          AW\)����@����D���B�𤿨��@��H�0���\\)BǸR                                    BxW�m,  T          AW����
@��\�B{p�B�uÿ��
@�=q�-��UBͮ                                    BxW�{�  �          AX����R@�ff�B�RaHB�� ��R@�ff�.=q�V��B�L�                                    BxW��x  �          A[�
@
=@   �TQ���B:�R@
=@�Q��H��aHB��\                                    BxW��  "          AY�?���@dz��P(���B���?���@�  �AG��{�RB���                                    BxW���  "          AX  ?�=q@^{�P  ��B�aH?�=q@�z��A���~B�33                                    BxW��j  �          AV�H?O\)@����J�H�B�p�?O\)@Ӆ�:=q�p�
B���                                    BxW��  %          AVff>L��@�G��I��\)B��)>L��@��H�8(��m
=B��{                                    BxW�Ӷ  	�          AV�\=�Q�@����I�ffB��=�Q�@�{�7��k=qB�p�                                    BxW��\  
�          AV=q=���@����Ip�B�B�p�=���@��H�8(��m�B�L�                                    BxW��  "          AV{�#�
@��I�L�B�=q�#�
@�
=�8���o\)B���                                    BxW���  "          AVff��ff@�Q��H(�33B��;�ff@�Q��6ff�ip�B�W
                                    BxW�N  �          AW33�0��@�=q�E���B��
�0��@���1��_=qB��=                                    BxW��  W          AW\)��Q�@�
=�G
=\)B��)��Q�@�{�4���dBƽq                                    BxW�+�  
W          AX��    @����H��.B��    @�G��6�\�e��B���                                    BxW�:@  
�          AYG�>�@�p��H��{B���>�@�z��6{�c�
B��H                                    BxW�H�  
�          AZ=q?(�@�  �I�k�B���?(�@�R�6ff�bB���                                    BxW�W�  
�          AY�>W
=@����H��Q�B�z�>W
=@�\)�5��b�\B���                                    BxW�f2  
�          AZ=q���@����IG�u�B��3���@�
=�6�\�c
=B�                                    BxW�t�  �          AZff=#�
@�\)�H  ��B�#�=#�
@���4���_p�B�aH                                    BxW��~  "          AZ=q�u@��\�H���fB�33�u@�Q��6{�b=qB��3                                    BxW��$  "          AZ�\��R@�G��I�
=B�p���R@�
=�6�\�b�
B��\                                    BxW���  �          AZ=q�h��@�  �H����BǸR�h��@���6ff�c{B�
=                                    BxW��p  T          AZ�\�333@�z��Hz�{B��Ϳ333@�G��5�aQ�B���                                    BxW��  �          AZ�H�Q�@��\�G33B�B��Q�@�ff�4(��]�B�.                                    BxW�̼  �          AZ�\�:�H@����E�B�uÿ:�H@�(��2=q�Z�B���                                    BxW��b  �          A[��
=@�  �D����B�(��
=A��0���W=qB�L�                                    BxW��  �          A[�
�:�H@��E�k�B���:�HA (��2=q�X��B�z�                                    BxW���  �          A[��@  @���E� B�ff�@  @�\)�2{�X��B��3                                    BxW�T  T          A\  ��R@��H�F�Hk�B�33��R@���3��Z�B�                                    BxW��  �          A\(��L��@����G\)��B��
�L��@�33�4Q��[��B��3                                    BxW�$�  �          A\z�E�@��
�G
={B��E�@��3�
�Z�B��                                    BxW�3F  T          A\�׿G�@���E�~B��G�Ap��2{�Wp�B���                                    BxW�A�  T          A\�׿@  @����F{�ffB�33�@  A ���2�R�X=qB���                                    BxW�P�  T          A]G��!G�@�G��D���{(�B�ff�!G�A���0���T{B���                                    BxW�_8  �          A\�Ϳz�@�p��EG��}Q�B�Ǯ�z�A�R�1�VffB�(�                                    BxW�m�  �          A]녿\)@���D���yp�B�  �\)A=q�0z��R�B���                                    BxW�|�  �          A^=q�+�@���D���y\)B��+�A=q�0���R��B�#�                                    BxW��*  �          A^�H�8Q�@�\)�D���xffB��R�8Q�A33�0���R
=B�Ǯ                                    BxW���  
�          A_33�=p�@θR�Ep��x�
B�#׿=p�A�R�1p��R��B��                                    BxW��v  "          A^�H�aG�@����Ep��yp�BÙ��aG�A�1�S�B���                                    BxW��  �          A_��O\)@�\)�C��s��B��ͿO\)A
�\�/33�N�B�                                    BxW���  �          A`Q�n{@���C��r�\BÅ�n{A��.�H�M  B�#�                                    BxW��h  
�          A`�ÿQ�@�(��C��q�
B��\�Q�A���/
=�L\)B���                                    BxW��  �          A`�׿!G�@ۅ�C��rp�B��\�!G�A(��/33�M{B�G�                                    BxW��  
�          A`�׿fff@ᙚ�A���nQ�B=�fffA�R�,���I=qB��                                    BxW� Z  T          A`�ÿ�=q@��>�R�h33Bĳ3��=qA
=�)G��C\)B�k�                                    BxW�   "          A`�׿n{@����@���lQ�B��
�n{A�
�+�
�G�\B���                                    BxW��  
�          Aa��c�
@߮�B�H�o��B�p��c�
A���.�\�K33B�p�                                    BxW�,L  T          AaG���  @��H�DQ��rp�BĞ���  A33�0Q��N�B��                                    BxW�:�  "          Aa��k�@�  �B�R�o��B��k�AG��.�\�KffB��)                                    BxW�I�  
�          Aa�h��@�ff�A���l33B�ff�h��AQ��-��H33B��                                    BxW�X>  �          Ab�\�z�H@��@���ip�B�33�z�HA�\�,  �E��B�=q                                    BxW�f�  
(          Aa���\@�
=�>�H�f�HBÊ=���\A�
�)��C=qB���                                    BxW�u�  �          Abff�}p�@�ff�?�
�g�B�.�}p�A��+
=�D33B�G�                                    BxW��0  
�          Ab�\�\(�@�z��@���i(�B�G��\(�A�\�,  �E��B��q                                    BxW���  "          Ac\)�u@�z��A���i�\B���uA�\�-��FffB���                                    BxW��|  T          Ac���{@�  �C
=�k�Bŀ ��{Az��/
=�I{B�{                                    BxW��"  �          Ac�����@�(��Ap��i{B�uÿ���A{�-G��FffB���                                    BxW���  �          Ac\)���@�33�A��j
=B�\���A���-��Gz�B�                                      BxW��n  
�          AdQ�u@�G��AG��gG�B�uÿuAQ��,���D�
B�Ǯ                                    BxW��  
�          Adz῀  @��Bff�iG�B�B���  A�R�.ff�G{B�ff                                    BxW��  �          AdQ쿈��@���C��l  BĽq����A(��0(��J  B��=                                    BxW��`  T          Ad(���p�@�p��DQ��m\)BǊ=��p�A�\�1��K��B�Ǯ                                    BxW�  
�          Ad�׿�33@�\)�D(��l�B���33A\)�0���K{B{                                    BxW��  
�          AeG��u@��A�f�B�k��uA���-�E  B���                                    BxW�%R  �          Ae���k�@��B=q�f��B��)�k�A���.ff�E��B�\)                                    BxW�3�  �          Ae���z�H@��A��f��B³3�z�HA���.=q�Ep�B�{                                    BxW�B�  �          Af�\�n{@����A���dz�B����n{A�H�-���Cz�B�=q                                    BxW�QD  
�          Af�H�n{@��R�@  �a�B�LͿn{A���+�
�@G�B�
=                                    BxW�_�  T          Af�R���\@�(��@z��b(�B½q���\AQ��,z��A�\B�8R                                    BxW�n�  T          Af�H��  @�{�@  �a33B�W
��  A���,  �@B��                                    BxW�}6  T          Ag
=�}p�AG��>�H�^B�  �}p�A
=�*�R�>�B��3                                    BxW���  T          Af�H�}p�A��>�R�^�HB��}p�A�\�*�R�>��B���                                    BxW���  �          Ag\)���A
=�=��\�RB�(����A(��)�<�HB��q                                    BxW��(  T          Ag���=qA=q�>�H�]��B����=qA��*�H�>�B��                                    BxW���  "          Ah(����A��>ff�\�B£׿��A���*ff�=  B�Q�                                    BxW��t  �          Ah�Ϳ�=qA���;
=�V�Bţ׿�=qA ���&�\�6�HB�                                      BxW��  	�          AiG����A\)�9��SG�BŅ���A#��%��4=qB���                                    BxW���  
�          Aj=q���A���=G��W�B�B����A!��(���8B���                                    BxW��f  "          Al  ��
=A
�R�=��VffBÀ ��
=A#
=�)p��7��B�8R                                    BxW�  T          Amp�����A=q�<���R�RBř�����A&ff�((��4=qB�{                                    BxW��  "          Ao33����A
=�>ff�S�B��Ϳ���A'33�)���4��B�k�                                    BxW�X  �          ApQ쿾�RA�R�<���O  B�p����RA*�\�'��0�B��f                                    BxW�,�  �          Aq녿�
=Ap��@(��R33B��)��
=A)p��+\)�4=qB�\)                                    BxW�;�  �          Ar{��(�A�
�AG��T  BƔ{��(�A'�
�,���6=qB��                                    BxW�JJ  �          Aq��G�A���9��I\)B�  ��G�A/�
�$z��+�B�\                                    BxW�X�  �          Ap�׿�{A"�\�/��;G�B�\��{A7�
�����B�                                      BxW�g�  �          Ar=q����A$  �0z��;
=B��Ϳ���A9G��{��
B�Ǯ                                    BxW�v<  
�          Ar�H���A{�5�A�HBɮ���A4  � (��%
=B��                                    BxW���  �          As
=��ffA ���4  �?33B�aH��ffA6{�=q�"z�B�                                      BxW���  "          As����A{�6�H�B��Bɨ����A3�
�!���&(�B��                                    BxW��.  �          As���A   �4���?�B��f��A5G��\)�#ffB�B�                                    BxW���  �          As33��z�A Q��4Q��?ffBɏ\��z�A5p���H�#G�B�{                                    BxW��z  �          As����HA"=q�2�H�={B��f���HA7
=�p��!(�B�u�                                    BxW��   �          At(����
A&{�0���9�Bǅ���
A:ff�
=��B�k�                                    BxW���  �          Aup���A'��0���8Q�B�녿�A;�
��H��B���                                    BxW��l  T          Atz��33A*{�,z��3�\B��)�33A=��R�33Bǣ�                                    BxW��  T          As33���A-���%��,
=B��)���A@Q���
�  Bʅ                                    BxW��  T          At(��\)A*=q�+
=�2{B��\)A=p��p��(�B�ff                                    BxW�^  T          Atz���RA&�R�/33�7{B�.��RA:ff�=q�\)Bɳ3                                    BxW�&  �          At����\A,���)p��/
=B����\A?���
��Bɨ�                                    BxW�4�  �          Au�
=A-���)���.�B�z��
=A@Q��(��\)B�.                                    BxW�CP  �          Av�R���A'
=�1p��8
=B��
���A:�\�����B�8R                                    BxW�Q�  "          Ax  ���A'
=�2�H�8B�ff���A:�R�ff��HB˸R                                    BxW�`�  �          Ax���A+
=�0���5
=B̸R�A>{��
�=qB�L�                                    BxW�oB  �          Ayp��   A/�
�,Q��.��B͔{�   AB=q�33�=qB�8R                                    BxW�}�  T          Ayp��*�HA4���%��&�BΔ{�*�HAF�\�z����B�L�                                    BxW���  
�          Ay��$z�A7��#\)�#�HB�8R�$z�AH����
�RB��                                    BxW��4  "          Ax���(��A6�R�#��$p�B�  �(��AG�
�ff�z�B��
                                    BxW���  �          Azff���HA,  �3��7{BȮ���HA>�R���{BƮ                                    BxW���  �          A{���G�A'\)�9��>�B�{��G�A:�R�&�\�%��B�#�                                    BxW��&  
�          A|  ���A*ff�7
=�:p�B�{���A=G��#\)�!��B��                                    BxW���  �          A}p����HA+33�8Q��:�\B��Ϳ��HA>{�$���"�B�Ǯ                                    BxW��r  "          A}G���z�A-G��6{�7�HB���z�A?�
�"ff���B�\                                    BxW��  �          A|Q���HA1��0���1��B�����HAC
=�����HB�G�                                    BxW��  �          A|���A0Q��1��3
=Bɀ �AB{�=q�33BǏ\                                    BxW�d  "          A|����A0z��2{�2�Bɽq��AB{��\�G�B���                                    BxW�
  
�          A|�����A6ff�+33�*�RB��
���AG33�\)�=qB�
=                                    BxW�-�  �          A|(���A8Q��'�
�&��B��)��AH���  ��RB�\                                    BxW�<V  �          A|���\)A;�
�$���"ffB���\)AK�
�z��ffB��                                    BxW�J�  �          A}��z�A6ff�,���+\)B�\�z�AF�H�G���B�.                                    BxW�Y�  �          A�z��"�\A ���Dz��Gz�B�ff�"�\A3��3\)�1
=B̀                                     BxW�hH  T          A�
=�=qA!p��E���G�
B��H�=qA4Q��4z��1z�B�(�                                    BxW�v�  �          A�33�\)A%p��B�\�CQ�B�{�\)A7�
�1��-(�B�u�                                    BxW���  �          A�p����A(���@Q��?�Bͣ����A;
=�.�H�)�HB�=q                                    BxW��:  �          A��
�\)A&�H�B�H�B�\B��f�\)A8���1�,��B�W
                                    BxW���  �          A��
���A)p��A��@{B͔{���A;\)�/�
�*ffB�33                                    BxW���  T          A�{��A*�H�@z��>��B̏\��A<���/\)�)p�B�Q�                                    BxW��,  T          A�ff�33A+
=�Ap��?�\B�B��33A<z��0Q��*33B�\                                    BxW���  �          A��R��
A-p��@(��==qB�\��
A>�R�/
=�(�B��                                    BxW��x  �          A��R�(�A,z��@���=�RB͊=�(�A=�/��(��B�L�                                    BxW��  �          A����'�A)��C��A=qB����'�A:ff�333�,��B�L�                                    BxW���  �          A�33�)��A)p��C�
�A(�B�{�)��A:�\�3��,�RB͔{                                    BxW�	j  "          A�p��)��A(Q��Ep��B�B�=q�)��A9���5G��.ffB͸R                                    BxW�  T          A�\)�'
=A(  �E���C�B��H�'
=A8���5���/
=B�aH                                    BxW�&�  �          A��
�G�A3��=���7�
B��H�G�AC�
�,���#B�                                    BxW�5\  �          A�  ��
A2�H�>�\�8�HB�\)��
AC
=�.{�%  B�z�                                    BxW�D  �          A�=q�   A1�?��9B�Q��   AA��/\)�&�B�G�                                    BxW�R�  �          A�Q��&ffA333�>{�7�B�#��&ffAC
=�-�$G�B�{                                    BxW�aN  "          A�z��   A333�>�H�8p�B���   AC
=�.�H�%33B�(�                                    BxW�o�  �          A����(�A5���>=q�6B�G��(�AE��.{�#��B�k�                                    BxW�~�  Y          A�\)�!�A8(��<z��3�B����!�AG��,Q��!  B���                                    BxW��@  
�          A�p���A7�
�=�5Q�B�����AG
=�-�"�B�=q                                    BxW���  "          A��(�A8  �?
=�6Q�Bɔ{�(�AG
=�/33�#�B���                                    BxW���  �          A�{�p�A;33�<���2��B�ff�p�AI��,��� �B��)                                    BxW��2  "          A�=q��\A:�R�=p��3�B�.��\AIp��-�!G�Bș�                                    BxW���  "          A��\�
=A9���?
=�4�B��
=AHQ��/��"�HB�aH                                    BxW��~  �          A��\�=qA:ff�>=q�3��B�\)�=qAH���.�H�"{BɸR                                    BxW��$  �          A������A;��=���2�RB˙����AJ{�.=q�!{B���                                    BxW���  T          A����!�A<z��<���1�RB�G��!�AJ�\�-� G�Bʣ�                                    BxW�p  �          A��H��A<Q��=��2(�B�\)��AJ=q�.{� �
B�Ǯ                                    BxW�  
�          A���#33A?��:=q�.=qB�
=�#33AM��+
=�(�B�z�                                    BxW��  T          A��
�p�AA���:ff�-\)B��p�AO
=�+\)�ffBɏ\                                    BxW�.b  �          A����(�A)��O��G�B��
�(�A8���Bff�7=qB���                                    BxW�=  T          A���(�A,Q��Mp��E33B͊=�(�A;
=�@(��4�B˙�                                    BxW�K�  T          A�  � ��A.�H�LQ��B�
B��)� ��A=G��?
=�2�\B���                                    BxW�ZT  �          A�ff�#�
A5��G��<G�Bͅ�#�
AC
=�:{�,(�B�                                    BxW�h�  �          A����#33A9��D���8p�B��)�#33AF�\�7\)�(p�B�8R                                    BxW�w�  T          A��H�'
=A9�D���7�RB�L��'
=AG33�733�'�Bˣ�                                    BxW��F  T          A����,(�A:�R�C�
�6�B����,(�AG�
�6ff�&��B�L�                                    BxW���  �          A����+�A;�
�B{�4Bͳ3�+�AH���4���%\)B��                                    BxW���  T          A����z�AK
=�3��$�BƊ=�z�AV�H�%p���BŊ=                                    BxW��8  T          A�z��G�AU��'���B��)��G�A`������=qB�B�                                    BxW���  T          A��Ϳ��AS\)�,z���\B��῅�A^ff�{�ffB��=                                    BxW�τ  �          A��ÿ��AS�
�,���Q�B��H���A^�R�ff�\)B�ff                                    BxW��*  
(          A�
=�z�HAV�H�)p��B�W
�z�HAap��33�	�B��                                    BxW���  �          A�G�����AZ=q�%G��
=B������AdQ��
=�p�B���                                    BxW��v  �          A�Q쿱�A[\)�&�H�z�B������Aep�����{B��                                    BxW�
  �          A�z`\)A]p��$���=qB�LͿ�\)Ag33��H�  B���                                    BxW��  �          A�zῷ
=A]G��$���=qB�Ǯ��
=Ag
=�
=�33B�L�                                    BxW�'h  T          A��H���HA]p��&�R�p�B�  ���HAg33�����\B��{                                    BxW�6  �          A�p���{A_��%���B�{��{Ai��  �{B��R                                    BxW�D�  "          A�G���ffAa��"�H�{B��\��ffAjff�G���\B�#�                                    BxW�SZ  �          A�\)���A`z��#
=�(�B�\)���Ai�������HB��
                                    BxW�b   "          A�  ����AdQ�� ���Q�B�������AmG��\)��Q�B�G�                                    BxW�p�  �          A��׿O\)Ag��
=�	�B�ff�O\)ApQ������\)B�(�                                    BxW�L  
�          A��R�p��Aip��z��
=B�aH�p��Aq��33��z�B��                                    BxW���  �          A��ÿp��AiG������B�\)�p��Aq���z����\B��                                    BxW���  �          A�
=�O\)Aip����=qB�Q�O\)Aq�������B��                                    BxW��>  T          A���:�HAjff����{B��q�:�HArff�(�����B��\                                    BxW���  �          A�G��Q�Ak��33�Q�B�aH�Q�As���\��z�B�(�                                    BxW�Ȋ  �          A���B�\Al���G���\B��H�B�\AtQ������G�B��3                                    BxW��0  :          A�\)�W
=Aj�H������B��\�W
=Ar�\�����(�B�\)                                    BxW���  �          A�\)�s33Aj�R����B�p��s33Ar=q�����ffB�33                                    BxW��|  �          A�����Ak��ff�B�B����Ar�H��\����B�                                      BxW�"  �          A�\)���HAf�H�!���B��\���HAn�\����B�33                                    BxW��  �          A�p����HAe�#\)�(�B������HAmG��(��33B�=q                                    BxW� n  T          A������HAg��!��
�RB�zῺ�HAo33�{���B�#�                                    BxW�/  �          A�(���Aj�R�=q�=qB���Aq�
=��G�B���                                    BxW�=�  �          A�G��\)Ao
=��\���
B�
=�\)Aup������Bę�                                    BxW�L`  T          A���ffAl�������B�.�ffAs������
Býq                                    BxW�[  "          A�=q���HAk��(��=qB�#׿��HArff�����ffB½q                                    BxW�i�  �          A����   Ak33�{��BÀ �   Aq�������B�{                                    BxW�xR  �          A�ff���Al�����Q�B�uÿ��As33�G���33B�{                                    BxW���  �          A�ff���RAk
=���\B�uÿ��RAqp���
��{B�
=                                    BxW���  
�          A��׿�Aj�\�\)��HB��ÿ�Ap�������
=B{                                    BxW��D  
�          A��ÿ޸RAg\)�%G��Q�B��R�޸RAm���
�(�B�W
                                    BxW���  �          A��H��ffAc
=�+�
���B�uÿ�ffAi���"�R�
��B��                                    BxW���  �          A�G���p�Abff�.�\�(�B��ÿ�p�Ah���%���G�B��                                    BxW��6  �          A�\)���Anff����ffB������AtQ��z���p�B�aH                                    BxW���  T          A�녿���Ah���(Q��
=B�W
����An�R����\B�{                                    BxW��  �          A��Ϳ�
=Aw33�33���
B�k���
=A|(��{��33B�.                                    BxW��(  �          A��Ϳ�
=AxQ��
�\��  B�aH��
=A}��p���B�(�                                    BxW�
�  T          A��H���HAs\)�33��  B��
���HAxz��
�\��  B��{                                    BxW�t  �          A��ÿ�(�AtQ������B����(�Ay��	p����B��{                                    BxW�(  �          A�33�33Atz��G����
B�G��33AyG�������B�                                      BxW�6�  �          A�(�����Aq�
=�{B��H����Av�R�
=��33B���                                    BxW�Ef  �          A�zῺ�HAp  �
=���B�#׿��HAt���\)����B��                                    BxW�T  T          A��R��=qAm��#
=�	{B�#׿�=qAr�H�����B��f                                    BxW�b�  �          A�=q��\)Ak��$  �
�RB��\��\)Ap������B�L�                                    BxW�qX  �          A�Q��Q�Am�� �����B����Q�Ar�\�� �
B��q                                    BxW��  T          A�z��Ap(��ff�  B��3��At���\)����B�u�                                    BxW���  �          A��R��(�AqG��p���
B���(�Au��\����B���                                    BxW��J  �          A��H���
As
=��H�Q�B�ff���
Aw33�(���{B�.                                    BxW���  �          A����(�At�����{B��Ϳ�(�Ax�������  B���                                    BxW���  �          A�\)��  Au��{���B�  ��  Ax������{B���                                    BxW��<  "          A��׿��
A~=q����B�=q���
A���� ����=qB��                                    BxW���  �          A�����Aj�\�(����
B�z���An�\�"�H�=qB�33                                    BxW��  "          A�33�"�\Ad���-��Q�B�.�"�\Ah���(z���B��
                                    BxW��.  �          A���	��Aa�2�H�Q�B�W
�	��Ae�-���{B�
=                                    BxW��  �          A���� ��A`���2{��B�G�� ��Adz��,����HB��                                    BxW�z  �          A��\�p�A^�R�3\)��B�\�p�Ab�\�.�\�Bǽq                                    BxW�!   �          A�����A]G��5����B��H��Aa��1G��=qBǏ\                                    BxW�/�  �          A��R�@��Av=q��
���B�#��@��Ax���
�H��B��f                                    BxW�>l  
�          A�
=�1�Az{�  ����B�33�1�A|���
=��(�B�                                      BxW�M  �          A�p��+�Ar�\���   B�
=�+�Aup��p����B���                                    BxW�[�  T          A���;�Ap(��  �  B�\�;�As
=�����B���                                    BxW�j^  
�          A���=p�Ak33�#33��HB��=p�An{�33�  Bʀ                                     BxW�y  �          A�G��L(�Af{�)��B��L(�Ah���%��ffB̽q                                    BxW���  �          A��
�U�A`���-����RBΨ��U�Ac\)�)��(�B�W
                                    BxW��P  �          A��
�j=qAS\)�?�
�$��B����j=qAV=q�<z��!G�Bҏ\                                    BxW���  �          A��e�Aa��/��\)BЅ�e�Ac��,Q���B�8R                                    BxW���  �          A�(�<#�
A�ff���
����B��<#�
A����(���G�B��                                    BxW��B  �          A��H=���A�G������33B�aH=���A���ʏ\��33B�aH                                    BxW���  �          A��\>.{A�\)��p���B��H>.{A�  ��ff��{B��f                                    BxW�ߎ  T          A���>��A��
���
��(�B�
=>��A�ff�������B�
=                                    BxW��4  �          A��>�(�A��H���H���\B�8R>�(�A�\)��z����B�=q                                    BxW���  �          A�G�>�
=A��H���
��33B�G�>�
=A�p���{��z�B�G�                                    BxW��  �          A��?
=qA����ff����B��?
=qA���������ffB��=                                    BxW�&  �          A�\)>�A�  �Ϯ����B�#�>�A�z��ʏ\���RB�#�                                    BxW�(�  �          A�G�<��
A��������RB��)<��
A�{������HB��)                                    BxW�7r  (          A�p��#�
A�=q������z�B���#�
A����ȣ�����B��                                    BxW�F  
�          A�p�=L��A����G���{B��=L��A�Q���p����HB��                                    BxW�T�  �          A���>��A��R���H���\B�>��A�
=��\)��B�                                    BxW�cd  �          A��?�A��R��ff����B��=?�A�����33��z�B��\                                    BxW�r
  �          A��>�Q�A��R��33���RB���>�Q�A�����Q����\B���                                    BxW���  �          A��>8Q�A��\�����=qB��
>8Q�A������H��ffB��
                                    BxW��V  �          A�=#�
A���
=��ffB��R=#�
A���������B��R                                    BxW���  �          A�G�=uA������
��33B���=uA����=q���B���                                    BxW���  �          A�Q�L��A�33��{����B�LͽL��A�G�������  B�L�                                    BxW��H  �          A�{>.{A�����p�����B��>.{A�����z����B��                                    BxW���  "          A�ff>��A�{���
��  B�Q�>��A�{��33���B�Q�                                    BxW�ؔ  �          A�Q�=���A�z���z����B�aH=���A�z���z����
B�aH                                    BxW��:  "          A�{=uA�����  ��z�B���=uA�����Q����RB���                                    BxW���  "          A���<��
A��H��\)��  B��H<��
A�����  ���\B��H                                    BxW��  �          A�\)=uA�������Q�B���=uA�����z����B���                                    BxW�,  �          A���=�G�A���������p�B�B�=�G�A��R��=q���\B�B�                                    BxW�!�  �          A��=�\)A�33������(�B��=�\)A�
=���R����B��                                   BxW�0x              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�?              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�M�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�\j              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�k              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�y�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��\              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��N              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�њ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��@              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�2              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�)~              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�8$              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�F�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�Up              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�d              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�r�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��b              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��T              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�ʠ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��F              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�8              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�"�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�1*              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�?�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�Nv              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�]              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�k�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�zh              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�æ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��L              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��>              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�*0              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�8�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�G|              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�V"              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�d�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�sn              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��`              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��R              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��D              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�#6              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�1�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�@�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�O(              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�]�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�lt              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�{              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��f  �          A�=q�uA�p�������RB���uA��R�   ��B��
                                    BxW��  
�          A�{��G�A�p���
=���B����G�A�����\��B�#�                                    BxW���  T          A�����ffA��ÿ������\B�W
��ffA�Q��{��\)B�\)                                    BxW��X  �          A��xQ�A�33�������B��f�xQ�A�z��{��
=B��                                    BxW���  �          A���L��A�
=��=q��G�B��
�L��A�ff�{��\)B��)                                    BxW��  �          A�p��aG�A��Ϳ���G�B�W
�aG�A�{��
��  B�\)                                    BxW��J  �          A�
=��33A�Q쿎{�XQ�B�� ��33A�� ���ÅB��=                                    BxW���  �          A��\��A��n{�5B�.��A�33��=q���HB�33                                    BxW��  �          A��H����A�(�����^�RB�ff����A����
��Q�B�k�                                    BxW�<  �          A��׿���A�{����^�RB��q����A����
����B�                                    BxW�*�  T          A��H��=qA�Q쿓33�`��B�����=qA�������HB���                                    BxW�9�  �          A�33�Y��A��R��G��tz�B�#׿Y��A�  �(���p�B�(�                                    BxW�H.  
�          A�Q쿔z�A����\���\B���z�A����{��ffB�\                                    BxW�V�  �          A��\�
=A�=q�z�H�<��B��{�
=A����Q���33B��{                                    BxW�ez  �          A��\��A�Q�L����HB��q��A��
��\���HB�                                    BxW�t   �          A�zᾸQ�A�Q�#�
��B�#׾�Q�A��
��\)���
B�#�                                    BxW���  T          A�{����A�  ��Q쿊=qB�����A��������G�B�                                    BxW��l  �          A��
����A���  �E�B��;���A�p����R�p  B���                                    BxW��  
Z          A�{�#�
A�{���
�uB���#�
A������}p�B�                                      BxW���  
�          A�Q�aG�A�=q�����fffB�L;aG�A�녿�ff�{�B�L�                                    BxW��^  �          A�  �W
=A�  �.{�   B�B��W
=A����
=�c33B�B�                                    BxW��  �          A�  �ǮA�녾u�:�HB�\)�ǮA�����G��s33B�\)                                    BxW�ڪ  
�          A�  �W
=A���z�c�
B�
=�W
=A�p������~�RB�\                                    BxW��P  �          A�  ����A�������fffB�(�����A�33��=q��  B�.                                    BxW���  �          A�
=�   A�
==�G�>���B���   A��Ϳn{�333B���                                    BxW��  �          A�G�����A�G�<#�
<�B�  ����A�
=����G�B�                                      BxW�B  T          A�=q�J=qA�{>��
?xQ�B��q�J=qA�녿:�H���B��q                                    BxW�#�  
Z          A�p��(�A�33?+�@�\B��3�(�A�G��\��33B��3                                    BxW�2�  �          A���
=qA�
=?\)?��HB�LͿ
=qA�
=���H��  B�L�                                    BxW�A4  T          A�녿\)A��
>���?h��B�\)�\)A���E��z�B�\)                                    BxW�O�  T          A��\)A���>�Q�?��B�ff�\)A���5�	��B�ff                                    BxW�^�  T          A�p��k�A��?Q�@{B��{�k�A�G�����J=qB��{                                    BxW�m&  
�          A�
=�\(�A���>�{?��\B�B��\(�A��R�=p����B�B�                                    BxW�{�  �          A��+�A���>�ff?�\)B�
=�+�A���#�
��
=B�
=                                    BxW��r  T          A��
��(�A��
����p�B�����(�A����(��l(�B���                                    BxW��  �          A�  �#�
A��>�{?��
B��)�#�
A��B�\�33B��)                                    BxW���  �          A�녿O\)A���?O\)@��B�녿O\)A�������h��B��f                                    BxW��d  �          A��
��RA�p�?���@[�B��R��RA��<��
=uB��R                                    BxW��
  "          A���\A���?0��@�
B�B��\A�
=��G���=qB�B�                                    BxW�Ӱ  �          A����A���?xQ�@9��B�p���A��þ.{�   B�p�                                    BxW��V  �          A�33�0��A��R?�\)@�33B�#׿0��A��>k�?.{B�#�                                    BxW���  �          A����A���?�=q@~{B�p���A�
=>8Q�?�B�k�                                    BxW���  "          A�
=��33A���?��@z�HB�{��33A���>#�
>��B�{                                    BxW�H  �          A��R��(�A�(�?˅@���B��{��(�A���>�(�?�ffB��{                                    BxW��  �          A�z��RA���?��@�{B�Ǯ��RA�=q?:�H@��B�                                    BxW�+�  �          A��Ϳ   A�{?�G�@���B���   A���?
=?�\B���                                    BxW�::  �          A�ff�\A�?�
=@�=qB�Q�\A�Q�?�?�ffB�L�                                    BxW�H�  "          A���ǮA�Q�@33@�
=B�\)�ǮA�33?���@[�B�W
                                    BxW�W�  �          A�  ���
A�\)?��H@��B�녾��
A��?�?���B��                                    BxW�f,  �          A�Q��A��?ٙ�@�z�B���A�=q?�?���B�                                    BxW�t�  �          A�(�<��
A��?��R@��B��)<��
A�{>���?h��B��)                                    BxW��x  
�          A�\)���A�
=?�@`  B��=���A�G�������B��=                                    BxW��  �          A�ff��Q�A��
?˅@���B�.��Q�A�Q�>Ǯ?�z�B�.                                    BxW���  "          A�ff�\)A�p�@�@ҏ\B��H�\)A�=q?z�H@<��B��H                                    BxW��j  
�          A�Q��G�A�\)@�@�(�B��׾�G�A�{?Tz�@�RB���                                    BxW��  T          A�ff>�Q�A��H@(��@��RB�Ǯ>�Q�A��?�
=@���B���                                    BxW�̶  �          A�z�?333A���@#�
@�
=B��R?333A�  ?���@���B�                                    BxW��\  �          A��\?��RA���@0��A��B�� ?��RA�?��@�(�B��=                                    BxW��  �          A�=q?k�A�(�@>�RA(�B�k�?k�A�\)?�G�@��B�u�                                    BxW���  �          A��?
=qA��
@AG�A�RB��3?
=qA��?�@��RB��q                                    BxW�N  �          A����
A�  @7
=A
�\B�����
A�33?У�@�{B��                                    BxW��  T          A�  >�  A�(�@=p�A33B�p�>�  A�\)?�p�@��RB�u�                                    BxW�$�  �          A��?\)A��@L(�A�RB��{?\)A�
=?��H@�p�B���                                    BxW�3@  �          A��
>L��A�@HQ�A�
B�Ǯ>L��A�
=?��@�\)B���                                    BxW�A�  T          A�      A�(�@:�HA��B���    A�\)?�
=@��\B���                                    BxW�P�  T          A��
>�p�A��@Q�A33B��q>�p�A���@�\@��B�                                    BxW�_2  T          A��>ǮA��@UA"{B���>ǮA���@ff@ʏ\B���                                    BxW�m�  T          A��?�A��@]p�A(  B�Ǯ?�A���@p�@�{B���                                    BxW�|~  �          A�p�<�A�G�@L(�A�B�Ǯ<�A���?�Q�@�z�B���                                    BxW$  T          A�33���
A�33@C�
A�B�����
A��\?�@��B��                                    BxW�  �          A��H=�\)A�Q�@\(�A(Q�B��\=�\)A��
@(�@�p�B��\                                    BxW¨p  �          A��ͽ#�
A�=q@\��A(��B�B��#�
A�@��@�{B�B�                                    BxW·  �          A�z�>W
=A��@hQ�A1�B��>W
=A�G�@�@�  B��3                                    BxW�ż  T          A��\>��A��
@a�A,��B�aH>��A�p�@G�@�p�B�ff                                    BxW��b  �          A��\>8Q�A�  @^{A*{B��H>8Q�A��@p�@׮B��f                                    BxW��  �          A�z�>\)A���@i��A2�HB�(�>\)A�33@Q�@���B�(�                                    BxW��  
�          A�z�>#�
A���@g�A1��B�>#�
A�G�@ff@�{B�                                    BxW� T  ~          A��\>.{A�@fffA0Q�B��>.{A�\)@�@�33B���                                    BxW��  "          A��R>�A�p�@uA<(�B�(�>�A�33@$z�@��HB�33                                    BxW��  �          A��H>B�\A�ff@XQ�A%G�B��
>B�\A��@ff@�z�B��)                                    BxW�,F  �          A��R���
A��R@Dz�A�B�zὣ�
A�{?��
@�p�B�z�                                    BxW�:�  �          A��׽L��A�33@!G�@�{B�Q�L��A�=q?�p�@o\)B�Q�                                    BxW�I�  
�          A��׽#�
A���@C33A�B�B��#�
A�  ?�G�@�33B�B�                                    BxW�X8  
�          A��H��\)A�\)@�G�AEB��q��\)A�G�@0  A=qB��R                                    BxW�f�  �          A�\)���
A�33@�33AT��B��ᾣ�
A�G�@Dz�AG�B���                                    BxW�u�  "          A�=��
A�(�@�=qAFffB��==��
A�{@1G�A�\B��=                                    BxWÄ*  �          A��<�A�=q@���AD  B���<�A�{@.{A  B���                                    BxWÒ�  "          A���L��A���@��AN�HB�=q�L��A��@<(�A�RB�8R                                    BxWáv  T          A����p�A�33@��A[�B�G���p�A�\)@L(�A33B�B�                                    BxWð  �          A�\)��A��@��HAG�
B�33��A��@2�\A�B�(�                                    BxWþ�  �          A�\)�(�A��@{�A?�
B�Ǯ�(�A�@'�@��RB��q                                    BxW��h  T          A�
=��RA��@���AD��B��
��RA�\)@-p�A(�B�Ǯ                                    BxW��  �          A���.{A�33@�AL��B�B��.{A�33@7�A  B�33                                    BxW��  �          A��ÿaG�A�ff@�Q�A]G�B���aG�A��\@Mp�A��B�p�                                    BxW��Z  \          A�
=�h��A��\@��RAZ�\B���h��A��R@I��AB���                                    BxW�   ~          A�\)�0��A�ff@�  Ahz�B�Q�0��A��R@\(�A'�B�=q                                    BxW��  T          A�G��aG�A��\@��Ab{B�� �aG�A���@S33A ��B�k�                                    BxW�%L  
�          A�G����HA��R@�ffAYB��\���HA��H@HQ�A��B�p�                                    BxW�3�  �          A�33�uA��\@���A^�RB���uA���@N�RAp�B��f                                    BxW�B�  
�          A�\)���A���@���AW
=B�uÿ��A�
=@E�AB�\)                                    BxW�Q>  �          A�G��˅A��@��AH��B��f�˅A��@2�\A�B�Ǯ                                    BxW�_�  �          A�33��A�p�@n{A5��B��H��A��@��@��B��q                                    BxW�n�  "          A�\)�
=qA��@j=qA2�\B�W
�
=qA�G�@�@�\B�.                                    BxW�}0  T          A��� ��A��@���AE��B�u�� ��A�
=@.{A(�B�L�                                    BxWċ�  T          A�33��A��\@�ALQ�B����A��\@6ffA
�HB��                                     BxWĚ|  "          A�G���=qA�ff@�(�AVffB�ff��=qA��\@C�
A��B�=q                                    BxWĩ"  T          A�33��Q�A�{@�(�Ab�HB���Q�A�ff@S33A!�B��H                                    BxWķ�  T          A�\)��{A���@�G�A^=qB�𤿎{A��H@Mp�AQ�B��
                                    BxW��n  "          A�G����A���@�ffAYB�uÿ��A���@G
=A�B�\)                                    BxW��  "          A�p��uA��@���AW33B��ÿuA�33@C�
A�B��)                                    BxW��  T          A�G��c�
A���@�p�AX(�B��=�c�
A��@E�A�B�u�                                    BxW��`  �          A�\)�Q�A��@{�A?�
B���Q�A��
@%�@�33B�
=                                    BxW�  �          A�G��z�HA�p�@�z�AJ=qB�\�z�HA�p�@333A  B���                                    BxW��  T          A����E�A�33@�{AX��B��ͿE�A�\)@FffA�\B��q                                    BxW�R  
�          A����k�A���@��Aap�B��q�k�A��@QG�A
=B���                                    BxW�,�  �          A���uA��R@�(�Ab�\B��ÿuA�
=@R�\A (�B��H                                    BxW�;�  "          A�G��c�
A�=q@���AiB��\�c�
A���@[�A'\)B�u�                                    BxW�JD  �          A�G��G�A��\@�33Aa��B��)�G�A���@P��A
=B�Ǯ                                    BxW�X�  �          A���&ffA�G�@�Q�A\(�B�\�&ffA�p�@J=qA��B�                                      BxW�g�  �          A���333A���@���A]��B�W
�333A�33@L(�A
=B�G�                                    BxW�v6  �          A���W
=A�ff@�33AmG�B�=q�W
=A��H@`��A*�RB�(�                                    BxWń�  
Z          A����^�RA�(�@�
=As33B�uÿ^�RA���@hQ�A0��B�\)                                    BxWœ�  T          A�G�����A�z�@��A��B�.����A�G�@�p�AK�B�
=                                    BxWŢ(  �          A���xQ�A��@�33Ay�B���xQ�A�=q@p��A7\)B�                                    BxWŰ�  "          A�p��O\)A��@�Q�AuG�B���O\)A�ff@j�HA2�RB�                                    BxWſt  T          A���
=A�(�@��RAr�RB��R�
=A���@g
=A0  B���                                    BxW��  "          A�\)��RA��@��A{33B�녿�RA�{@q�A8z�B��
                                    BxW���  T          A�G��p��A��\@�
=A��\B��ÿp��A�\)@�z�AJ�\B��
                                    BxW��f  �          A�\)�\(�A�(�@�A��
B�p��\(�A��@��AT��B�W
                                    BxW��  T          A�G�����A�33@�
=A�\)B��쿈��A�Q�@��Ad(�B��                                    BxW��  
�          A�33�\)A���@��A��B��=�\)A��@���AD��B�u�                                    BxW�X  
�          A��H��{A�(�@�p�A���B��q��{A�\)@��Ao
=B��\                                    BxW�%�  �          A���p��A�@���A�\)B��ÿp��A���@��\AT(�B��)                                    BxW�4�  �          A�ff����A��@ÅA�B�aH����A���@���Al��B�.                                    BxW�CJ  �          A�(���=qA�G�@�=qA���B��)��=qA�z�@���Ak�B���                                    BxW�Q�  �          A�ff��Q�A�33@�G�A��B�8R��Q�A�  @�
=AO�
B�
=                                    BxW�`�  
�          A�Q��ffA��@�ffA��
B���ffA���@���Aep�B�=q                                    BxW�o<  �          A�z���A���@��A���B�uÿ��A��
@���AS\)B�=q                                    BxW�}�  �          A��R�\A���@�(�A��B�Ǯ�\A��@�=qA`��B��\                                    BxWƌ�  T          A�ff��{A�=q@��A�{B�녿�{A�G�@�Q�A]�B��                                    BxWƛ.  �          A��׿�
=A�G�@�ffA�
=B����
=A�(�@�z�AW�
B�aH                                    BxWƩ�  T          A��ͿW
=A�
=@���A��
B�ff�W
=A�{@��HAaG�B�L�                                    BxWƸz  �          A��þ���A�Q�@�G�A���B��{����A���@�\)At��B��=                                    BxW��   �          A��׿   A��@˅A��
B�=q�   A�
=@��AyG�B�(�                                    BxW���  �          A��׾�
=A�@�33A��B��R��
=A�
=@���Ay�B���                                    BxW��l  T          A�z�z�A�33@ϮA�\)B��z�A��\@�ffA�=qB��                                    BxW��  �          A�ff�\A���@���A��B�uþ\A�(�@��A��\B�ff                                    BxW��  �          A�ff�   A�=q@�Q�A��\B�G��   A��
@�
=A�p�B�33                                    BxW�^  T          A�=q����A�\)@�  A��B��ᾙ��A�
=@�\)A�{B��                                    BxW�  �          A�=q=���A���@�ffA�ffB�\)=���A��\@�{A�\)B�aH                                    BxW�-�  
�          A�{=uA��@���A��
B���=uA��
@�z�A���B���                                    BxW�<P  T          A�=q<#�
A�ff@�G�A���B��<#�
A�=q@���A��
B��                                    BxW�J�  "          A�=q=�A��@�{A��RB�.=�A��
@�A��
B�33                                    BxW�Y�  �          A�Q�>�ffA�ff@�\A��B���>�ffA�=q@�=qA��RB�\                                    BxW�hB  
�          A�z�>�ffA��
@�p�A��RB�>�ffA�p�@���A��B��                                    BxW�v�  "          A�ff>�  A�G�@θRA���B�ff>�  A��\@�A�
B�p�                                    BxWǅ�  �          A�{��\)A�(�@�\A�{B�uý�\)A�  @\A�G�B�u�                                    BxWǔ4  �          A��
�\)A�@��
A�p�B��f�\)A���@�(�A��RB��H                                    BxWǢ�  	�          A��
=A�ff@��RA�ffB��ÿ
=A�z�@�
=A�B��
                                    BxWǱ�  �          A����\A�Q�@�
=A��HB�k���\A�Q�@�\)A�=qB�Q�                                    BxW��&  
�          A����G�A�p�@�ffA��B�  ��G�A���@�
=A�z�B��                                    BxW���  �          A���G�A�=q@���A�Q�B���G�A�Q�@љ�A�B��q                                    BxW��r  �          A�����Q�A��@�G�A�
=B��{��Q�A�  @�=qA��\B��\                                    BxW��  
�          A�>B�\A��@�Q�A�G�B��q>B�\A��@�  A���B�Ǯ                                    BxW���  
�          A����
A���@�A�ffB��=���
A��H@�=qA�  B��=                                    BxW�	d  
�          A��
>�
=A�(�@�
=A�\)B�33>�
=A��@�\)A�
=B�G�                                    BxW�
  T          A��
>aG�A��@�A�
=B��=>aG�A�\)@�ffA��RB��{                                    BxW�&�  	�          A�����A��@��AƸRB�33���A�@�z�A�z�B��                                    BxW�5V  	7          A�\)�aG�A�p�Ap�A�=qB�{�aG�A�@���A�{B��H                                    BxW�C�  	�          A�>�(�A�  @�  A�=qB�#�>�(�A��@���A�(�B�8R                                    BxW�R�            A�G�>�A���@�  A��B�#�>�A�z�@�G�A���B�(�                                    BxW�aH  O          A��׾���A���@��A�=qB�=q����A��@˅A�=qB�.                                    BxW�o�  �          A��׾���A�ff@�z�A��B�������A�(�@�{A��B��{                                    BxW�~�   �          A��þL��A�33@���A�(�B�W
�L��A�33@ӅA�Q�B�L�                                    BxWȍ:              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWț�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWȪ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWȹ,              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��x              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�j              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�.\              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�=              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�K�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�ZN              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�h�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�w�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWɆ@              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWɔ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWɣ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWɲ2              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��~              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��$              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��p              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�
              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�'b              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�6              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�D�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�ST              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�a�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�p�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�F              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWʍ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWʜ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWʫ8              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWʹ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�Ȅ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��*              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��v              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW� h              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�/              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�=�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�LZ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�[   X          A�G��
�HA^�\A7�B��B��
�HAh��A*{B\)B���                                    BxW�i�  	�          A�����
Af=qA/
=B�HB�G���
Ap  A!p�B�BÞ�                                    BxW�xL  "          A��
��\Ag\)A.�\B{B�{��\Ap��A ��B�B�p�                                    BxWˆ�  T          A��
���HAip�A+�
Bz�B�Q���HAr�HA{BffB¸R                                    BxW˕�  T          A�녿�(�Af{A0��B{BÞ���(�Ao�
A#33B{B���                                    BxWˤ>  �          A�  ���HAc
=A5��B��B��R���HAl��A(Q�B�
B�#�                                    BxW˲�  
�          A�  ��Ab{A6�HB�B�uÿ�Al(�A)B33B��f                                    BxW���  �          A�녿��HA_�A9��B
=B��ÿ��HAi��A,��B33B�\)                                    BxW��0  T          A����A^=qA:�HBffB�uÿ��AhQ�A.{B��B��H                                    BxW���  S          A��˅A]�A<Q�B!  B���˅Ag33A/�BG�B��=                                    BxW��|  �          A����A_
=A8��BB�{��Ah��A,Q�B�B�p�                                    BxW��"  T          A�녿�\)A_�A:{B�\B�=q��\)Aip�A-p�B  B�                                    BxW�
�  T          A����G�AT��AC�
B)�B��쿡G�A_\)A7�
BffB�Q�                                    BxW�n  
�          A�(��#33AW33A:ffB!��B�z��#33Aa�A.ffBQ�Bȏ\                                    BxW�(  �          A���z�HAZ�RA9p�B �B�(��z�HAdz�A-G�B�B���                                    BxW�6�  T          A�����(�AXQ�A;33B"�
B�aH��(�Ab{A/\)B��B��)                                    BxW�E`  "          A��
��p�AV=qA>�\B%��B�����p�A`  A2�HBB�\                                    BxW�T  �          A����(�AT  A@��B(Q�B��׿�(�A]�A5�B33B�{                                    BxW�b�  �          A�����AL  AA�B,  B�\���AU�A6{B {B�(�                                    BxW�qR  T          A�z��ffAQ�A8z�B$(�B�B��ffAZ�\A-�BG�B�z�                                    BxW��  �          A�ff�(�AMp�A<(�B(\)B�aH�(�AV�HA1G�B��BƊ=                                    BxW̎�  �          A��
�,(�AE�AAG�B/�B̙��,(�AN�HA6�HB#�B�z�                                    BxW̝D  �          A���Dz�AE�A?�
B-ffB����Dz�AN�RA5p�B!��BθR                                    BxW̫�  �          A�  �I��AD(�AAG�B.��B��)�I��AMA7
=B#G�BϏ\                                    BxW̺�  �          A���B�\ADQ�A@(�B.{B��
�B�\AM�A5�B"��BΙ�                                    BxW��6  ,          A�(����AO�
A8z�B$�B��f���AX��A-B(�B�
=                                    BxW���  
�          A�p��\)AN=qA8z�B%z�BǸR�\)AW33A-�B33B��f                                    BxW��  
(          A�  ����AS�
A4z�B �Bî����A\z�A)Bp�B�{                                    BxW��(  
�          A���z�AW\)A1G�B�B��
��z�A_�
A&�\BG�B�u�                                    BxW��  
�          A�{?z�Aap�A\)B  B�L�?z�Ah��A(�B��B�u�                                    BxW�t  	�          A�>��HA`Q�A�B�B�>��HAg�
A��B�\B�#�                                    BxW�!  �          A�=q?5AaA33B�B�=q?5Ai�A(�B��B�k�                                    BxW�/�  "          A�p�?@  A_�A�B�HB���?@  Af�HA��B��B�                                    BxW�>f  
Z          A�(�?=p�A[33A ��B  B���?=p�Ab�\AffB(�B�                                    BxW�M  �          A���?��AQA'�B�B���?��AYp�Ap�B33B�                                    BxW�[�  �          A�  ?��RALz�A,  B��B��?��RATQ�A"ffB(�B�W
                                    BxW�jX  
�          A��?}p�AN{A)�BB�.?}p�AU��A Q�B33B��                                     BxW�x�            A�
=?^�RAPQ�A$��BG�B�W
?^�RAW�A33B�
B���                                    BxW͇�  �          A��
?k�AS33A#�
B  B�?k�AZffA{B��B�G�                                    BxW͖J  �          A�{?O\)AU�A ��B�B�{?O\)A\��A33B	ffB�L�                                    BxWͤ�  �          A��R?�G�Aep�A�
A�=qB���?�G�Ak\)A��A��
B�8R                                    BxWͳ�  T          A���?���Ab�HAz�B�B��?���Ah��AffA�33B�W
                                    BxW��<  �          A�
=?��RA^=qA33B
=B���?��RAd��AG�B
=B��                                    BxW���  �          A�=q?���AhQ�A�B(�B�p�?���An�\A	��A�\B�                                    BxW�߈  �          A���?\Ah��Ap�BQ�B�
=?\An�RA\)A�
=B�\)                                    BxW��.  �          A��\?��
Ab{A33BG�B��?��
Ah��Ap�B�\B��                                    BxW���  �          A���?Tz�A\z�A#33B��B��?Tz�Ac33AB(�B�Q�                                    BxW�z  �          A�?��A[�
A$Q�BB��=?��Ab�\A
=B	=qB���                                    BxW�   �          A�33?�z�ALQ�A5p�B%��B���?�z�AS�A,��B�B�                                      BxW�(�  �          A��?��\AO�A.ffB�
B���?��\AV�\A%B�B�G�                                    BxW�7l  �          A�(�>aG�AP��A,��Bz�B�
=>aG�AW�
A$z�B33B��                                    BxW�F  �          A�z�?��
AX  A%BB�=q?��
A^ffA��B��B��                                     BxW�T�  �          A�\)?�p�Ac�
A�HBG�B�.?�p�Ai��AA��\B��                                    BxW�c^  �          A�\)?��RAj{A�A�ffB�� ?��RAo\)A=qA��B��
                                    BxW�r  �          A��@@  Ao�@���A���B�W
@@  At(�@��A�\)B�Ǯ                                    BxW΀�  �          A�(��
=qA2�\AS
=BF��B���
=qA:ffAL  B>(�B�G�                                    BxWΏP  �          A����W�@�(�A{\)B}\)B�\�W�@�
=Av�HBup�B�#�                                    BxWΝ�  �          A�\)��=q@�RAy��Bx(�B��H��=q@�G�Au�Bp��B�(�                                    BxWά�  �          A�\)����@�z�A�  B���B������@�\)A|(�B|�\B��                                    BxWλB  �          A����Q�@���A�(�B��{B��\��Q�@�33A|��B~33B���                                    BxW���  �          A�{���@�\)A33B�8RB�L����@��A{�B�B�=                                    BxW�؎  �          A�=q�p��@���A�{B��B����p��@�\)A�z�B��=B�G�                                    BxW��4  �          A�{�`  @���A��\B���B�33�`  @��HA�
=B�33B��H                                    BxW���  �          A�Q��a�@�
=A���B�Q�B���a�@�G�A�(�B�B��                                    BxW��  �          A��H�Z=q@��A���B��{B�8R�Z=q@�p�A�33B�Q�B�k�                                    BxW�&  �          A����a�@�p�A�\)B�=qB����a�@�\)A��B�  B�Q�                                    BxW�!�  �          A�ff�o\)@��A��HB�aHCn�o\)@�G�A��B�p�B���                                    BxW�0r  �          A�=q�w
=@tz�A��B��)C^��w
=@��A�{B��C��                                    BxW�?  �          A����@�\A�ffB�p�C5����@5�A�B�� CǮ                                    BxW�M�  �          A�z���z�?�
=A�G�B�� C  ��z�@{A���B�C33                                    BxW�\d  �          A�����Q�?(��A�\)B�aHC,)��Q�?�
=A��B��3C&�                                    BxW�k
  �          A�����=q�\)Az�HB�{CI����=q��  A{�
B�G�CEs3                                    BxW�y�  �          A�\)�����
At  B�u�CA�����Q�Atz�B�#�C<�3                                    BxWψV  �          A��R��ff��{Aup�B���CB
=��ff�fffAu�B�� C=c�                                    BxWϖ�  �          A�=q����L��Av=qB�\)C6+����>��Av=qB�W
C133                                    BxWϥ�  �          Ay���ff>��Ao\)B��RC.�q��ff?W
=Ao
=B�B�C)B�                                    BxWϴH  �          A~�\���H@p�Aq��B��C����H@8��Apz�B�#�CǮ                                    BxW���  �          A|z��(�@�=qAX  Br  B�aH�(�@�{AT��Bk�
B��
                                    BxW�є  �          A|z῜(�A1�A1��B3(�B��=��(�A6�RA,��B,�HB�8R                                    BxW��:  �          A�
=?&ffAM�A&=qB�B�33?&ffARffA ��BQ�B�Q�                                    BxW���  �          A�(�?J=qAg�A�RA���B�Ǯ?J=qAk33Az�A���B��H                                    BxW���  �          A�\)?�  Ak�@�(�Aڣ�B�8R?�  An�H@�AθRB�Q�                                    BxW�,  �          A�
=?�z�At��@��A�(�B�\)?�z�Ax  @��A�z�B�u�                                    BxW��  �          A�z�?}p�Aj=qA
=qA�B�8R?}p�AmAQ�A�Q�B�W
                                    BxW�)x  �          A�p�?��Ab=qA�A���B��R?��Ae��A{A�p�B��f                                    BxW�8  �          A���?�\)Adz�A  A��\B�\?�\)Ag�
AffAB�=q                                    BxW�F�  �          A�{?���A\��A��B�\B���?���A`��A\)B33B�                                    BxW�Uj  �          A��\?�Q�AP(�A"ffB��B�\?�Q�AS�
A��B\)B�\)                                    BxW�d  �          A�  ?�G�AZ�RA�A�{B�G�?�G�A]��@��A��
B�z�                                    BxW�r�  �          A�ff?��RA^�\@��A��B��q?��RAaG�@�Aܣ�B��                                    BxWЁ\  �          A|��@5Ad��@�=qA��RB��{@5Ag
=@�Q�A�
=B�Ǯ                                    BxWА  �          A{33@�A^=q@ٙ�A�Q�B���@�A`z�@�Q�A���B�#�                                    BxWО�  �          Ay��@-p�A`(�@�Q�A�p�B��@-p�Ab=q@�
=A�=qB�G�                                    BxWЭN  �          Ax��?�z�Ag
=@�=qA�ffB���?�z�Ah��@���A�G�B��H                                    BxWл�  �          Ax  ?��Aa�@�33A�\)B�?��Ac�@�=qA�z�B��H                                    BxW�ʚ  �          Ay@
=Am�@��A��
B���@
=Anff@�=qArffB��f                                    BxW��@  �          Ax��@L(�AW�@�p�A�(�B�\)@L(�AY@�p�A�  B���                                    BxW���  �          Ay�@ ��AG�A�B��B��@ ��AJ=qA(�B	�\B��f                                    BxW���  �          Au>\AI�Az�B��B��=>\AK�A��B�B��{                                    BxW�2  �          Aw33?��RAQ�AG�A���B���?��RAS33@��A���B��                                    BxW��  �          Am@J=qA=A��B33B�8R@J=qA@  A��B�\B��=                                    BxW�"~  �          Ak\)@�=qA>�\@��
A�\)B��@�=qA@z�@�A�\B�p�                                    BxW�1$  �          Amp�@-p�A[33@�  A�  B���@-p�A\(�@s33ArffB��3                                    BxW�?�  �          Al��@>{Ag
=?��@�ffB��q@>{Ag\)?���@�(�B�                                    BxW�Np  �          Alz�@.�RAf{?�\)@��B�� @.�RAf�\?�z�@�
=B��                                    BxW�]  �          Ak�
@'
=A]��@���A�
=B���@'
=A^ff@��HA���B��R                                    BxW�k�  �          Ak�@4z�A\(�@���A�(�B���@4z�A\��@��
A�Q�B��                                    BxW�zb  �          Al  @8Q�Aa�@S33AO
=B��H@8Q�Aa@G�AC�
B��                                    BxWщ  �          Ajff@/\)A\��@}p�A{�B��@/\)A]p�@s33Ap��B���                                    BxWї�  �          Ah��@
=qAM��@�=qA��HB��H@
=qAN�R@�p�A�B���                                    BxWѦT  �          Ah��@%AP(�@��A�{B�u�@%AQ�@���A�33B��{                                    BxWѴ�  �          Aj{@�RAM��@��
A�G�B�W
@�RAN�\@ϮAҸRB�p�                                    BxW�à  �          Ai@
=AM�@���A���B�k�@
=AO
=@��A�ffB��                                     BxW��F  �          Ah  @AW�
@�
=A�(�B�Q�@AX��@�33A�{B�aH                                    BxW���  �          Al��?���A[33@�=qA���B�Ǯ?���A[�
@�ffA��B��
                                    BxW��  �          Aqp�?�33A?
=A�\Bz�B�  ?�33A@Q�A�B��B�\                                    BxW��8  �          Ar�\?��
A(��A-�B6G�B��H?��
A*=qA+�
B4��B���                                    BxW��  �          Aq�?L��A#33A1��B=Q�B�\?L��A$z�A0z�B;B�#�                                    BxW��  
�          Av�H>aG�A�AG�BW��B�(�>aG�A=qAF�RBV�B�33                                    BxW�**  �          AxQ�>�z�A(�AC�
BP�B�z�>�z�AG�AB�HBO33B��                                     BxW�8�  �          Az�\?h��A��AB�RBL{B�aH?h��A�AA�BJ�B�u�                                    BxW�Gv  �          A33���@�ffAb�RBx�B�L;��@�Q�Ab{Bw�HB�=q                                    BxW�V  �          A{�
?ǮA8(�A&�RB'�B��=?ǮA8��A%�B&��B��{                                    BxW�d�  �          A{�?G�A3�A.�HB0�\B�\?G�A4Q�A.=qB/��B�{                                    BxW�sh  �          A{\)?p��A+\)A6�HB;
=B��?p��A+�
A6ffB:ffB���                                    BxW҂  �          A}�?��A.�HA6ffB8G�B�{?��A/\)A5�B7B��                                    BxWҐ�  �          A|(�?&ffA+�
A8  B;��B�\?&ffA,(�A7�B;=qB�{                                    BxWҟZ  �          Az�R>�=qA&ffA;�BA��B�\>�=qA&�\A;\)BA\)B�\                                    BxWҮ   �          Azff>.{A\)AG33BS
=B���>.{A�AG
=BR�B���                                    BxWҼ�  �          A{33=#�
A�AO33B^\)B�z�=#�
A�AO33B^ffB�z�                                    BxW��L  �          A{\)��\)Az�AS
=Bdz�B��f��\)AQ�AS33Bd��B��f                                    BxW���  
�          A{���p�A�HATQ�Bf�B�
=��p�A�\ATz�Bfp�B�
=                                    BxW��  �          A|(��   @��
AZffBo��B�G��   @��HAZ�\BpffB�Q�                                    BxW��>  �          A|�þ�@�A]Bu�B�L;�@�Q�A^{Bv{B�Q�                                    BxW��  �          Azff��\)@�{AY�Br�B�33��\)@���AZ=qBr�
B�8R                                    BxW��  �          AxQ쾅�@��HA[\)BwG�B�{���@�G�A[�
Bx�B��                                    BxW�#0  �          Avff�#�
@�33A\��B~Q�B��3�#�
@�G�A]�BQ�B��R                                    BxW�1�  �          Ax(��L��@�\)A_�B��B�aH�L��@��A`  B��B�k�                                    BxW�@|  �          Av�\���@ə�A`��B���B������@�
=Aa��B�B�B��3                                    BxW�O"  �          Aw��#�
@��
Ae�B�k�B�\�#�
@���Ae��B��B��                                    BxW�]�  �          Az{�aG�@�p�Ag�B�u�B�=q�aG�@�=qAh(�B�=qB�L�                                    BxW�ln  �          A}>k�@��HAhQ�B�ǮB�>k�@�\)Ah��B���B��                                    BxW�{  �          A\)=�@�\)As\)B�u�B��=�@��As�
B�\)B�
=                                    BxWӉ�  T          A�  >���@�  Aq��B��{B�ff>���@��
ArffB��\B�=q                                    BxWӘ`  �          A}녿.{@�p�Ar�HB��HBƊ=�.{@���As�B��B�.                                    BxWӧ  �          A|�þ��@�33Aqp�B��B�����@�ffAr{B�
=B�\                                    BxWӵ�  �          A����@�p�Ak�B�Q�B�=q���@���Alz�B��B��                                     BxW��R  �          A�{��@�33Al��B�(�B��=��@�{AmB�k�B��\                                    BxW���  �          A��ͽu@�p�AmB��B�33�u@��An�HB�G�B�8R                                    BxW��  �          A��=�\)@�G�Am��B��B���=�\)@�33An�HB��=B�Ǯ                                    BxW��D  �          A���?�@��HAm�B�B�B�G�?�@���Ao
=B��qB��                                    BxW���  �          A~�H?Tz�@�\)Ah��B���B�p�?Tz�@���Aj{B�.B���                                    BxW��  �          A~{?��A
�RAR{B_B��\?��A�
AT  Bb�B�
=                                    BxW�6  �          A~=q@��Az�AP��B]
=B��
@��A	p�AS
=B`Q�B�8R                                    BxW�*�  �          A�{@�A�
AX(�Bfz�B�p�@�A z�AZ=qBi�HB���                                    BxW�9�  �          A�  ?��
@陚AaBw�HB���?��
@�=qAc�B{p�B���                                    BxW�H(  �          A\)?�p�@��Aj=qB�k�B�{?�p�@�=qAk�
B�B�B���                                    BxW�V�  �          A~�\?�G�@��An=qB���B�L�?�G�@���Ao�B��B��\                                    BxW�et  �          A|  @ ��@��
Ah��B�aHB���@ ��@��Aj�\B�Q�B��
                                    BxW�t  �          Ay�@��@��HAb=qB��B�=q@��@�=qAd  B��B��\                                    BxWԂ�  T          A{
=@�
@�  Aap�B��B��)@�
@�\)Ac\)B��HB�G�                                    BxWԑf  �          A|(�@(�@�  Ad��B�G�B�\)@(�@�
=Af�RB�k�B���                                    BxWԠ  �          A|z�?�@�(�Ah(�B��
B���?�@��\Aj{B�{B�                                    BxWԮ�  �          A~�R?���@�ffAk�B�z�B�33?���@�z�Am��B�ǮB�(�                                    BxWԽX  �          A|��@Q�@ǮAe��B���B�Q�@Q�@�p�Ag�B�
=B�z�                                    BxW���  �          A|  @
�H@�(�Aep�B�8RB�\@
�H@���Ag�B���B�\                                    BxW�ڤ  �          A}�?�Q�@�33An=qB�G�B�B�?�Q�@�Q�Ap  B���B���                                    BxW��J  �          A|Q�?�{@��\An�\B�B�B�#�?�{@�\)ApQ�B��
B�k�                                    BxW���  �          A{\)?��
@�=qAmG�B�ǮB�� ?��
@��RAo
=B�aHB�z�                                    BxW��  �          A{�?�\)@�p�Aip�B���B��?�\)@���Ak�B��qB�
=                                    BxW�<  �          A|z�?�z�@�\)Ak33B�k�B�B�?�z�@�33Amp�B�8RB��f                                    BxW�#�  �          Az{?Ǯ@��AiG�B�ǮB��H?Ǯ@���Ak�B��B��=                                    BxW�2�  �          Ax��?�
=@��
Ag
=B���B�=q?�
=@�
=Aip�B��B��                                    BxW�A.  �          AyG�?���@�
=Ag33B��B��q?���@�=qAi��B�8RB�(�                                    BxW�O�  �          Ay��?�ff@��Ag
=B��\B�k�?�ff@�z�Ai��B�B��H                                    BxW�^z  T          Ay��?�Q�@���Af=qB��{B�=q?�Q�@�\)Ah��B��
B��=                                    BxW�m   �          Ay?�G�@�p�Ac�B�z�B�33?�G�@��AfffB��qB���                                    BxW�{�  �          Ay��?��@�AeB�{B���?��@�  Ahz�B�u�B���                                    BxWՊl  �          Aw�>�@�Al��B���B��>�@~{An�RB�33B�L�                                    BxWՙ  �          Au�<�@��HAk
=B��B�B�<�@w�Am�B��RB�(�                                    BxWէ�  T          Au�����@�p�AfffB�ffB��q����@�ffAh��B�{B���                                    BxWն^  �          Au����G�@���Ac\)B��{B��3��G�@��Af=qB�\)B�u�                                    BxW��  �          Au�?(��@�{Ag
=B�(�B�k�?(��@�ffAiB���B��                                    BxW�Ӫ  �          Au�?J=q@�G�Ac�B�L�B�� ?J=q@���Af�\B�33B�{                                    BxW��P  �          AuG�?Q�@�(�Aa�B�p�B�(�?Q�@�(�Ae�B�ffB��3                                    BxW���  �          Au�?�@���Ab�HB�� B��q?�@���Af{B��{B�                                    BxW���  �          Au�?J=q@��AbffB�B�u�?J=q@�G�Ae��B�#�B���                                    BxW�B  �          AuG�?Tz�@�
=Ac33B��RB��?Tz�@�{Af�\B��B��)                                    BxW��  �          At  ?�33@��
AaB�aHB��?�33@��\Ad��B��\B��                                    BxW�+�  �          At��?}p�@�Q�Ae�B�  B��R?}p�@��RAh(�B�Q�B�G�                                    BxW�:4  �          Au��?�=q@���Ac�
B�B�Q�?�=q@��HAg33B�ffB�                                      BxW�H�  �          Aw
=?�{@�G�A^{B��B��3?�{@�\)Ab{B�ffB���                                    BxW�W�  �          AxQ�?�
=@�ffA\��B{(�B��q?�
=@�z�Aa�B��B�                                    BxW�f&  
�          Ax��?ٙ�@θRA`��B��B�� ?ٙ�@�(�Ad��B��B��R                                    BxW�t�  �          Aw33?�
=@���Ac33B���B�  ?�
=@�p�Af�HB�(�B�\                                    BxWփr  �          At��?�z�@�Q�Ad��B��HB�B�?�z�@���Ah  B�aHB�\                                    BxW֒  �          At(�?�ff@��Adz�B���B���?�ff@�G�Ag�
B�\)B�ff                                    BxW֠�  �          As�
?�G�@�33Ab�\B���B���?�G�@�
=Af{B�ffB�33                                    BxW֯d  
�          As�
?�G�@�  Ad  B�
=B���?�G�@��Ag\)B��B���                                    BxW־
  �          As�?�{@�(�Aa�B�B�B���?�{@��Ad��B�G�B�u�                                    BxW�̰  �          Aqp�?���@�{Ag
=B��\B�Ǯ?���@aG�AiB���B�aH                                    BxW��V  �          ApQ�?���@�p�A^=qB���B��?���@���Aa�B��B�=q                                    BxW���  �          Ap��?���@�=qAbffB��{B���?���@���AeB��B��{                                    BxW���  �          Ar{?���@��HA^�\B���B���?���@��Ab�RB���B��f                                    BxW�H  T          Aqp�?J=q@��A\z�B�33B��?J=q@�(�A`��B��qB�G�                                    BxW��  �          Ar{?E�@�Q�AfffB��RB��{?E�@s33AiB�G�B�                                      BxW�$�  T          A{\)?�@��\Ao�B�8RB���?�@dz�Ar�RB�k�B|�                                    BxW�3:  �          A}�@(�@��RAi��B��B�\@(�@��RAm�B��fB�ff                                    BxW�A�  �          A~{@(�@�Aj{B�33B�p�@(�@�p�AnffB��\B~Q�                                    BxW�P�  �          A|��?�G�@�=qAk�B��{B�p�?�G�@���Ao�
B�aHB�z�                                    BxW�_,  �          A}G�?�z�@��Aj{B�8RB�?�z�@��\An�RB��B�
=                                    BxW�m�  "          A|��@
=@�ffAip�B��qB�Q�@
=@��An{B�p�B�p�                                    BxW�|x  �          A{�?��H@��Ah��B��B�#�?��H@�Q�Amp�B��qB�8R                                    BxW׋  �          A{�@33@��HAh��B�� B�@33@���AmG�B�W
B���                                    BxWי�  �          Az=q@�@���Af�HB��3B��@�@��HAk�B���B�ff                                    BxWרj  �          Az�R@�@�\)Ac�
B�k�B�aH@�@��Ai�B�z�B�8R                                    BxW׷  �          Az=q?��@��AhQ�B��{B�8R?��@���Al��B��fB���                                    BxW�Ŷ  �          Az{?�\@��
Ae�B�B��?�\@�G�Ak
=B�\B�W
                                    BxW��\  T          A{�@33@�(�Ac
=B��fB�aH@33@�G�Ah��B�#�B��                                    BxW��  �          A|��@AG�@�
=A_\)Bx�B��=@AG�@�(�AeG�B�u�B{33                                    BxW��  �          A|z�@K�@�(�A]G�Bu�B�u�@K�@�G�Ac�B��RBy{                                    BxW� N  �          A}G�@.{@ҏ\Ab{B}  B�(�@.{@��RAh  B��)B�(�                                    BxW��  �          A}�@�@��Ac�B�\)B��)@�@�Aip�B�B���                                    BxW��  �          A|Q�@�\@���A^�HBx{B���@�\@��Aep�B�B�8R                                    BxW�,@  �          Ax��@!�@�(�AY�Bsz�B���@!�@�Q�A_�
B�u�B�\                                    BxW�:�  �          Ax��@��@�\)A\z�BzffB�@��@��HAb�RB�  B�\)                                    BxW�I�  �          Aw�?�{@�(�A[\)Bz{B��q?�{@��Aa�B��B�z�                                    BxW�X2  �          Av�H@��@�(�AUG�BzQ�B3Q�@��@�  AY�B�#�B�
                                    BxW�f�  �          AuG�@qG�@�  A`z�B��{BH=q@qG�@e�Ad��B�\B-�                                    BxW�u~  �          Au��@�  @{�A_�
B�k�BQ�@�  @?\)Ac�B��HB �
                                    BxW؄$  �          Av=q@�
=@\��Ab=qB��B  @�
=@\)Aep�B���A�33                                    BxWؒ�  �          Av=q@333@J=qAlQ�B��BA�H@333@	��Ao33B�p�BG�                                    BxWءp  �          At��@�Q�@dz�Ad(�B�aHB��@�Q�@%Ag�B���A�{                                    BxWذ  �          At��@�
=@��AP  Bj�RB@�
=@i��AT��Bs33A뙚                                    BxWؾ�  T          Au�@���@��RAM�Bc��B	33@���@s33AQ�Bl
=A���                                    BxW��b  �          Aup�@�G�@��ADz�BT��B {@�G�@���AJ�\B^�\B=q                                    BxW��  �          At  @�33@�G�A?
=BM�
B��@�33@�p�AEG�BW�RB�                                    BxW��  �          As�@�ff@���AH  B\�B)�@�ff@��AN=qBfB��                                    BxW��T  �          At��@��@�ffAW�Bv�RB7�@��@��RA]�B�B                                      BxW��  �          At��@�=q@��A\  B��B<Q�@�=q@x��Aa�B���B!�                                    BxW��  �          Ar�\@�{@�=qAX��B~
=B33@�{@S�
A]G�B�
=B{                                    BxW�%F  �          As�
@��@�\)AY�B}z�B�@��@L��A^ffB���A���                                    BxW�3�  �          Au�@��H@��AU�Bs�
BG�@��H@N{AZffB|�HAޣ�                                    BxW�B�  �          Aup�@��@���A]��B�(�Bff@��@E�Aa�B�33A���                                    BxW�Q8  T          As�@�33@G�A]�B��)A��
@�33@�\Aa�B��A��                                    BxW�_�  �          Atz�?�z�@�
=AbffB���B�=q?�z�@��Ahz�B�ffB�G�                                    BxW�n�  �          AtQ�?�\)@��
Aa��B���B��f?�\)@�  Ag�
B��B�p�                                    BxW�}*  �          At��?�ff@�ffAc
=B�u�B�W
?�ff@��AiG�B��B��\                                    BxWً�  �          At(�?\(�@�  A^{B�{B�L�?\(�@��
AeG�B�#�B��f                                    BxWٚv  �          AtQ�?���@�Q�Aap�B���B�?���@��Ah(�B��HB��                                    BxW٩  �          As
=>��@�p�AQp�BnG�B�>��@��HAZ�RB�p�B��)                                    BxWٷ�  �          At      @���AQG�Bm
=B�
=    @�AZ�RB�
B�
=                                    BxW��h  �          Au����@�(�AR�RBlG�B�����@�Q�A\z�B=qB��3                                    BxW��  �          As����
@�z�AO33BiQ�B�uÿ��
@���AX��B|(�B�k�                                    BxW��  �          At  ��  @���AXz�Bz�\B�  ��  @��
A`��B��\B֨�                                    BxW��Z  �          As
=���A�ALz�Be�B�ff���@޸RAV�\Bx�RB�G�                                    BxW�   T          Ar�H��
=A�AK�
BdffB��ῷ
=@�  AV=qBw�B��                                    BxW��  T          As��k�@�33ARffBo�B���k�@�{A\(�B�aHB�L�                                    BxW�L  T          Azff��@�{A[33BuB�#׾�@�
=Ad��B��
B��=                                    BxW�,�  �          A~{���@�ffA[\)Bn�B�(����@�
=Ae��B�L�B��f                                    BxW�;�  �          A\)��(�@��HA`Q�Bv33B��=��(�@�=qAj=qB�8RB��)                                    BxW�J>  �          A�    @�Q�Aap�Bw�RB�    @�\)Ak\)B��B�                                    BxW�X�  �          A\)>8Q�@�Q�AeG�B�RB�>8Q�@�{AnffB�(�B�Q�                                    BxW�g�  �          A\)>��@��
Ab=qBy�B�>��@��Al  B�33B�33                                    BxW�v0  �          A\)?G�@��Ae��B�G�B�(�?G�@�=qAn�RB��B��                                    BxWڄ�  �          A�>�  @�  A`z�BwffB�(�>�  @�Aj�\B�B�B�W
                                    BxWړ|  �          A
=����Ap�AX(�Bg�
Bƽq����@ᙚAc�B|��B�{                                    BxWڢ"  �          A
=���
AG�AZ�HBl�B���
@أ�Af{B��B�L�                                    BxWڰ�  �          A�>��
A(�AW�Bf��B��q>��
@�ffAc\)B|p�B���                                    BxWڿn  �          A�?��\A��AJ{BS{B�Ǯ?��\A��AW�Bh�RB���                                    BxW��  �          A���>�@�G�Aep�B��HB��=>�@�(�AnffB���B�=q                                    BxW�ܺ  �          A��\�#�
A  A9�BV��B�8R�#�
@��HAEp�Bl�B��\                                    BxW��`  �          A��\>�A-A<(�B=�B���>�A�\AL(�BSp�B��                                     BxW��  �          A�z�?��
@�(�A`(�Bt�
B���?��
@�
=Ak
=B��=B�=q                                    BxW��  �          A���@�G�A`��Bz�B�(���@��
Ak33B�W
B��H                                    BxW�R  �          A
=��@�
=Ad��B�{B�uü�@�Q�An�HB�z�B��{                                    BxW�%�  �          A�@!�@���As�B�(�Bh=q@!�@ ��Ax��B�33B3=q                                    BxW�4�  �          A�@P��@333Av=qB���B"z�@P��?�p�Ayp�B���A��                                    BxW�CD  �          A~�R@��@b�\At��B���B`(�@��?�(�Ayp�B��=B=q                                    BxW�Q�  �          A\)?���@y��Av�\B�\B��{?���@�
A{�B���Bp                                    BxW�`�  �          A�z�@��\@!�AmB�33A�33@��\?}p�Ap��B��RA ��                                    BxW�o6  
�          A�Q�@�p�@G�As\)B�  A��@�p�>�(�Aup�B���@�G�                                    BxW�}�  �          A�@8Q�?�(�AyG�B�ǮB	z�@8Q�>���A{33B��@љ�                                    BxWی�  �          A�p�@�R@���AuB�aHB|p�@�R@0��A{�
B��\BL=q                                    BxWۛ(  �          A���?��
@�33Ar�HB��{B��?��
@��RA{\)B�.B�
=                                    BxW۩�  �          A�Q�?��@ȣ�Ao�B��3B�ff?��@�z�Ax��B��{B�u�                                    BxW۸t  �          A�=q?˅@���Ahz�B|��B��H?˅@���As\)B�k�B���                                    BxW��  �          A�z�?��@�p�Af�RBx��B�{?��@�=qAr=qB�p�B��                                    BxW���  �          A��?\@�33Ah��Bz�HB��3?\@�\)AtQ�B���B�\)                                    BxW��f  �          A�p�?�z�@��
Ak�B�B��\?�z�@��RAv�RB���B�                                    BxW��  �          A�p�?�\)@�\Ag�
Bx{B��=?�\)@�As�
B�p�B�\                                    BxW��  �          A�\)?��H@�Q�Ah��By��B���?��H@�33At��B�W
B��{                                    BxW�X  �          A�p�?˅@��Ah(�Bw�HB�L�?˅@�z�At(�B�k�B��)                                    BxW��  �          A�G�?z�H@�Ai��B{Q�B���?z�H@��Aup�B�k�B���                                    BxW�-�  T          A���?��@�  Ai�By��B���?��@���AuG�B��=B�\                                    BxW�<J  �          A���?W
=@�=qAk\)B}��B��
?W
=@�33Aw
=B�B��
                                    BxW�J�  �          A���?O\)@�
=Al  B{B�B�?O\)@��Aw�B��{B�B�                                    BxW�Y�  �          A�=q?}p�@�Q�Ai�Bw
=B�p�?}p�@���Au�B���B�Q�                                    BxW�h<  �          A�=q?s33@�
=AiG�Bw�RB���?s33@�\)Av{B�
=B��                                    BxW�v�  �          A�Q�?c�
@�(�AhQ�Bu\)B�\?c�
@�z�Aup�B���B�k�                                    BxW܅�  �          A�{?\(�A��A^�HBf33B��?\(�@��
Am�B��B�L�                                    BxWܔ.  �          A��?�G�A	�A`  BhB���?�G�@��An�RB��
B�aH                                    BxWܢ�  �          A��?���@���Ag�Bup�B��H?���@��At��B�{B���                                    BxWܱz  �          A��\?�G�@�ffAl  B{z�B���?�G�@�(�Axz�B�8RB���                                    BxW��   �          A�{?��@�Al��B  B���?��@��HAx��B�  B���                                    BxW���  �          A��?�p�@θRAq�B��B�?�p�@��HA|  B�aHB��3                                    BxW��l  �          A�z�?�(�@�
=As33B��\B��f?�(�@�=qA}p�B���B�\                                    BxW��  T          A�Q�?�
=@�z�Ao\)B�L�B���?�
=@�  A{
=B���B���                                    BxW���  �          A���?�33@�Al��B{z�B��\?�33@��Ay��B��B��)                                    BxW�	^  �          A��?���@��HAx��B��=B�p�?���@g�A�
=B��RB{�H                                    BxW�  �          A�33@��@��A}B��\B�p�@��@#33A�ffB���BD�                                    BxW�&�  �          A��@.�R@�  A~�RB�(�B^�\@.�R?�(�A�Q�B���B33                                    BxW�5P  �          A���@:=q@3�
A�z�B��3B0{@:=q?E�A�Q�B�
=Al(�                                    BxW�C�  �          A��R?��\A z�Ae��Bq�HB�{?��\@���At(�B�G�B��R                                    BxW�R�  �          A���>�A	G�Ab�HBk�B��)>�@�ffArffB�33B�#�                                    BxW�aB  �          A�z�>Ǯ@�G�Am��BG�B��>Ǯ@��\Az�\B�\)B��                                    BxW�o�  �          A��\>��@ǮAu��B��RB�=q>��@��RA�=qB��=B��f                                    BxW�~�  �          A�=q��z�@�
=A}�B�u�B�\)��z�@Q�A�z�B�L�B���                                    BxWݍ4  �          A�ff@L(�@�\)AiBy33B��)@L(�@�Q�AvffB���Bf�                                    BxWݛ�  �          A�{@��@�z�An�\B�Q�B�k�@��@�(�AzffB�.Bz�
                                    BxWݪ�  T          A�=q@Q�@��An�HB�k�B�\@Q�@���Az�\B��=B��R                                    BxWݹ&  �          A�{����@�\)Af�\Bs�B��R����@�  Aup�B�(�B���                                    BxW���  �          A�?���@��Ap��B�G�B��H?���@�  A|��B�G�B�k�                                    BxW��r  �          A��?.{@�Ap  B�B�L�?.{@��A|(�B�
=B��                                    BxW��  �          A��?5@�G�Aq��B��B�(�?5@�ffA|��B��)B��H                                    BxW��  �          A��?B�\@��HAt��B��HB�33?B�\@n{A33B�=qB��                                    BxW�d  �          A�G�>�(�@��
Aq�B�=qB�=q>�(�@�  A}��B���B�ff                                    BxW�
  �          A�>L��@��AnffB��=B�>L��@��A{�B�B�B�aH                                    BxW��  �          A�>\)@���Amp�B��=B���>\)@�G�Az�HB�W
B��H                                    BxW�.V  �          A�=�\)@�\)Aj�RB{�B���=�\)@�(�Ax��B��)B��\                                    BxW�<�  �          A��=u@�z�Ai�By\)B�{=u@�G�Aw�
B���B��R                                    BxW�K�  �          A�p��?\)A	�A[33Ba�Bڅ�?\)@�=qAl(�BB���                                    BxW�ZH  �          A����O\)@�\)A`(�Bi�B�#��O\)@�{Ao�
B�p�B�33                                    BxW�h�  �          A����1G�@��Ac
=Bn�HB�\�1G�@�\)ArffB�k�B癚                                    BxW�w�  �          A�  ��  @���Ai��Bz��B��
��  @�Q�Ax  B�(�B�                                    BxWކ:  �          A����@ڏ\Ao�B��B�B���@�(�A|��B�L�B�Q�                                    BxWޔ�  �          A�  �Q�@���AmB�
=B�LͿQ�@��\A{\)B�W
B�B�                                    BxWޣ�  �          A��
>�Q�@�G�Av�\B��RB��)>�Q�@aG�A���B�33B�B�                                    BxW޲,  �          A��?�z�@��\Aw
=B�� B��\?�z�@R�\A��RB��=B�33                                    BxW���  �          A�?���@�ffAt��B��fB�?���@j=qA�(�B�(�B��                                    BxW��x  �          A�p�?O\)@�  Ar�RB��
B�33?O\)@~{A~�RB�ffB��f                                    BxW��  �          A�33?G�@�(�Atz�B��\B���?G�@e�A�B�.B�L�                                    BxW���  �          A�\)@G�?�(�A~�RB���B#�R@G�����A�Q�B���C���                                    BxW��j  �          A�
=@e���\A~�RB�p�C���@e��7
=Az�RB�k�C��                                    BxW�
  T          A��@*�H@A~=qB���B%�@*�H�L��A�z�B�#�C�}q                                    BxW��  T          A��?�\)@��RAy�B�8RB��3?�\)@4z�A�\)B��qB��                                    BxW�'\  T          A�Q�?��R@�=qAy�B��B���?��R@;�A��B�{B|=q                                    BxW�6  �          A��\?��@�  A{33B��B���?��@$z�A�=qB��qBe                                    BxW�D�  �          A��\?��@�(�A~�RB�(�B�
=?��?�A��HB��B&�                                    BxW�SN  �          A��?�(�@�p�A�(�B��B�8R?�(�?�Q�A�B�\)B1�
                                    BxW�a�  �          A�(�@y��?L��A�=qB�� A8��@y���ٙ�A��B��\C�8R                                    BxW�p�  �          A�  @0��?���A�p�B�A�z�@0�׿s33A��
B��C���                                    BxW�@  �          A�Q�@.�R?�p�A�  B��RA�@.�R���A�=qB��)C�"�                                    BxWߍ�  �          A��H@*=q@p�A��B���B��@*=q����A�
=B�ffC�h�                                    BxWߜ�  �          A�ff@\)@%A���B��)B8�@\)=L��A���B���?���                                    BxW߫2  �          A���@Q�@%A�{B��BI��@Q�<#�
A��B�8R>Ǯ                                    BxW߹�  �          A�
=?�Q�@+�A�ffB�{BX(�?�Q�=��
A�(�B��{@=q                                    BxW��~  T          A�ff?�(�@:�HA���B�\)Bn{?�(�>���A���B��)A,��                                    BxW��$  
�          A�z�@ff@	��A�=qB�B6��@ff��
=A�\)B��fC�\)                                    BxW���  �          A��@�?���A}�B�k�B*�
@����A�B�Q�C�!H                                    BxW��p  �          A�
=@(�?�{A~ffB���Bz�@(��s33A33B�  C�<)                                    BxW�  �          A�z�@�?�{A|z�B���B�R@��0��A}�B���C��{                                    BxW��  �          A�=q@z�@
�HA{\)B�#�B,@z�ǮA}��B�#�C�C�                                    BxW� b  �          A�  @�H@p�Az�\B�\)B)ff@�H��Q�A|��B�ffC��                                    BxW�/  �          A�
@?�
=A{33B�#�B�@�&ffA|��B��C�>�                                    BxW�=�  �          A�{@
�H@33A\)B���B-
=@
�H�
=A���B�\C�q�                                    BxW�LT  �          A��R@�@�A�ffB�=qBC�R@녾�{A��B��\C�J=                                    BxW�Z�  �          A���?��
@9��A|  B��{Bi�R?��
>�=qA�(�B�#�A
{                                    BxW�i�  �          A�
=?У�@G�A{�
B��
By��?У�>��A�Q�B���A�G�                                    BxW�xF  T          A��?�ff@eA~�HB�{B��?�ff?c�
A�z�B��B	�R                                    BxW���  �          A���@�@(�A��
B�z�BD�@�����A�G�B�#�C��q                                    BxW���  �          A�(�@
=@
=A��
B�(�B3�R@
=�#�
A��HB���C���                                    BxW�8  �          A�>�׿��
A�G�B�C�C�>���n�RA|(�B�33C��{                                    BxW��  �          A�\)��\)��=qA���B�Q�C����\)�q�A{33B��C�p�                                    BxW���  �          A��
@Z�H@��\A���B�k�BH
=@Z�H?��A�z�B��A���                                    BxW��*  �          A�ff@^�R@��A��B�p�BJ��@^�R?�Q�A�
=B�z�A���                                    BxW���  �          A�=q@N�R@W
=A���B�8RB8�@N�R?�A���B��
AG�                                    BxW��v  �          A�{@|(�@^{A�\)B�
=B%�\@|(�?&ffA�=qB��
A                                    BxW��  �          A��\@��@>�RA�(�B��B��@��>\)A�Q�B�B�?�33                                    BxW�
�  �          A��H@��H@;�A��B��HB{@��H=��
A��
B��
?�ff                                    BxW�h  �          A�G�@�Q�@E�A�Q�B�W
A���@�Q�>��A���B�=q@3�
                                    BxW�(  �          A��@�Q�@N�RA��HB�ffB{@�Q�>�p�A�p�B���@���                                    BxW�6�  �          A�ff@�=q@"�\A��HB�Q�A��@�=q��Q�A�ffB�G�C��                                     BxW�EZ  �          A�p�@�  @/\)A�33B��A��@�  ��A�
=B�\C�<)                                    BxW�T   �          A�z�@�p�@3�
A�(�B��A��H@�p��uA�{B���C���                                    BxW�b�  �          A�=q@��
@	��A��B�B�A��@��
�=p�A��\B��RC�9�                                    BxW�qL  �          A���@��
@33A��RB���A��@��
�
=A��B���C���                                    BxW��  T          A�z�@��H@G�A�Q�B��HA��@��H�(�A��B���C�q                                    BxW᎘  �          A���@Ǯ@
=A��\B�p�A��
@Ǯ�J=qA��B��\C�c�                                    BxW�>  �          A�33@ƸR@A�
=B�A���@ƸR�Tz�A��B���C�.                                    BxW��  �          A�p�@�=q?�p�A���B�L�A���@�=q�s33A��B��C��
                                    BxWẊ  �          A�\)@���@ffA�p�B�ffA��\@��ÿ(�A��RB�L�C�!H                                    BxW��0  �          A��
@���@��A}��B��A�z�@��þ�A�Q�B��C��                                    BxW���  �          A���@���?�\)A�B�B�A��H@�����{A�Q�B���C�Ф                                    BxW��|  �          A��@O\)?�p�A�{B��A�=q@O\)��G�A��B�\)C�Ǯ                                    BxW��"  
�          A��>.{>�Q�A�\)B�B�B�Q�>.{�,��A��B�W
C��                                    BxW��  �          A����>8Q�A��B��C�)����9��A���B��
C���                                    BxW�n  �          A��>��>�G�A��
B���B��>���)��A�=qB�C��)                                    BxW�!  �          A�>8Q�?^�RA��B��fB�#�>8Q���RA��\B��{C�Y�                                    BxW�/�  �          A�>\?B�\A��B�33B}�>\�
=A�Q�B��{C��                                    BxW�>`  �          A�{?��?333A��B��A�?����HA�=qB�#�C�                                    BxW�M  �          A���@	��>aG�A��B�B�@�p�@	���9��A��B�=qC�>�                                    BxW�[�  �          A��\@�
��A�G�B�8RC��q@�
�J=qA��HB��C��                                    BxW�jR  �          A��R?��H���RA�B�u�C�t{?��H�]p�A��HB��3C���                                    BxW�x�  �          A�{?��;�(�A�p�B���C��=?����dz�A�Q�B�  C��                                    BxW⇞  �          A�\)?��ÿ
=qA��HB�33C�Ǯ?����k�A���B��HC���                                    BxW�D  �          A���?�G��(��A�33B�C�w
?�G��s�
A�B��3C�ff                                    BxW��  �          A�p�>�{�@  A�33B�Q�C�%>�{�z�HA���B��RC�z�                                    BxWⳐ  �          A��?�׾���A�Q�B��qC�?���c�
A�G�B�
=C��                                    BxW��6  �          A���@��?�Q�A���B��{A���@���ٙ�A�p�B���C��                                    BxW���  �          A�
=@u�?�33A���B�W
A�ff@u����
A��HB��C�"�                                    BxW�߂  �          A�\)@���?�  A���B��{A�G�@��ÿ�Q�A���B�{C���                                    BxW��(  �          A��@tz�?��A���B��A�  @tz�ǮA�
=B��RC��)                                    BxW���  �          A��@k�?��A�=qB��RA��\@k���Q�A�B�
=C��                                    BxW�t  �          A�z�@@��>W
=A�ffB��@y��@@���G�A�{B���C��R                                    BxW�  �          A�\)@�G�@
=A�ffB���Aܣ�@�G���(�A��B��C���                                    BxW�(�  �          A�\)@��R@�RA�  B�=qAߙ�@��R����A��HB�\C��\                                    BxW�7f  �          A�G�@��
?ǮA���B��RA�=q@��
���A���B�#�C�E                                    BxW�F  �          A�\)@�=q?G�A�\)B�z�A.ff@�=q�$z�A��B��C���                                    BxW�T�  �          A�\)@��?�A�ffB��@�  @���3�
A��\B��HC�O\                                    BxW�cX  T          A��@��H>�(�A��B���@��\@��H�:=qA��B�8RC��H                                    BxW�q�  �          A��@�p�>�
=A��B��\@�
=@�p��<��A�
=B��{C��                                    BxW〤  �          A���@���aG�A�
=B��C��\@���eA��B��C�j=                                    BxW�J  �          A�@�(��&ffA��RB���C�  @�(���Q�A��HB���C���                                    BxW��  �          A�  @�z�h��A���B���C�s3@�z�����A��\B��3C���                                    BxW㬖  �          A�=q@��
��\)A�(�B�p�C�q@��
��\)A�33B�k�C��                                    BxW�<  �          A�Q�@�녿��
A��\B�\C���@������A��B�(�C���                                    BxW���  �          A��
@�\)�У�A���B�u�C�]q@�\)���RA~�RB�G�C�B�                                    BxW�؈  �          A�Q�@�����A��HB���C���@�����A�B�.C�H�                                    BxW��.  �          A�=q@��Ϳ��A�p�B��\C�t{@�����Q�A�ffB�ǮC��)                                    BxW���  �          A��@����RA���B���C�,�@������Ax  By�HC�4{                                    BxW�z  �          A��
@����3�
A�
=B�=qC�@ @����ÅAw\)Bx�RC��R                                    BxW�   T          A��@��׿�z�A���B���C�Z�@������HA�p�B�� C��H                                    BxW�!�  
�          A���@��H�33A�Q�B�� C���@��H��p�A{\)B�.C�~�                                    BxW�0l  �          A��H@�(���A��B�=qC��)@�(���Q�A33B��\C��f                                    BxW�?  �          A��H@�{���
A��\B�8RC�
@�{��  A��
B��C�y�                                    BxW�M�  �          A��@z�H��33A��B�p�C�%@z�H����A�Q�B��HC�N                                    BxW�\^  �          A��@vff���RA�B�33C��@vff��  A���B��C�~�                                    BxW�k  �          A��@~�R��
=A��B�p�C��H@~�R��ffA��\B���C�%                                    BxW�y�  �          A�p�@qG���A�  B���C�~�@qG�����A�=qB��C��\                                    BxW�P  �          A���@|(��\A�B��RC�C�@|(��|(�A�{B��C��                                     BxW��  �          A��@�  ��  A��B�\C�j=@�  ����A��RB�G�C��=                                    BxW䥜  �          A�G�@�\)��\)A���B��HC���@�\)���A��
B��{C��                                    BxW�B  �          A�\)@�p��z�HA��\B��HC��
@�p�����A��
B�ffC�!H                                    BxW���  �          A��\@�=q�uA�ffB�\C�U�@�=q����A��B��fC���                                    BxW�ю  �          A���@�(��:�HA�p�B�.C��@�(���=qA�
=B��C��R                                    BxW��4  �          A�@��H�G�A��RB�\C�.@��H���A�=qB�z�C�:�                                    BxW���  �          A��@�  �xQ�A�p�B�#�C��R@�  ���\A���B�(�C�o\                                    BxW���  b          A�  @���n{A�33B�.C�
=@����G�A�z�B��{C�\                                    BxW�&            A���@���fffA�\)B�C��=@������A���B��qC�K�                                    BxW��  �          A��@|�Ϳ�
=A�
=B��=C��{@|������A��
B�\C��q                                    BxW�)r  �          A�z�@|�Ϳ��A�ffB�
=C�T{@|�����A���B��)C�*=                                    BxW�8  T          A�p�@xQ쿼(�A�\)B�ffC���@xQ����A��B��qC��)                                    BxW�F�  T          A�G�@`�׿���A��B���C��@`����  A��B�  C�B�                                    BxW�Ud  �          A��
@|(���Q�A��B���C�h�@|(���33A�G�B�Q�C�33                                    BxW�d
  "          A���@��׿�{A�B��HC��@�����G�A�{B�ǮC��{                                    BxW�r�  �          A�{@����  A��\B�  C�y�@�����A�
=B�z�C���                                    BxW�V  
�          A���@�33�h��A��
B�#�C���@�33��=qA���B��C�\)                                    BxW��  �          A��
@�G����A�(�B�ffC��)@�G���
=A�
=B��qC���                                    BxW垢  T          A��@��ÿ��HA�{B���C�z�@�����(�A�(�B��qC�w
                                    BxW�H  	�          A�(�@�33��
=A���B���C�(�@�33��33A�B�ffC���                                    BxW��  �          A�  @�G���  A�G�B�� C�n@�G����A}B�C�}q                                    BxW�ʔ  T          A��R@�  ��A���B�u�C���@�  ���A|(�B~�C���                                    BxW��:  
(          A���@\)�`  A�B�ǮC�e@\)��z�AxQ�Bx��C���                                    BxW���  �          A�Q�@�=q�QG�A��B�\)C��{@�=q��{Ayp�Bz�
C�0�                                    BxW���  
�          A��@tz��I��A�B���C�>�@tz���=qAyG�B}p�C���                                    BxW�,  
�          A��R@��
�   A�33B��3C�Z�@��
��Az�\B�=qC��
                                    BxW��  "          A�\)@����0  A�B�u�C��
@�����ffAz�RB�=qC�                                      BxW�"x  �          A��@����5�A�=qB�Q�C�h�@����љ�A{\)B�RC��                                    BxW�1  T          A�{@�Q��Dz�A�  B�p�C�Ff@�Q�����Ay�B|��C�K�                                    BxW�?�  �          A�{@�
=�Y��A���B�\)C���@�
=����Ar�RBvz�C�~�                                    BxW�Nj  �          A�ff@q��c33A�{B�(�C�e@q���RAtQ�Bw��C��3                                    BxW�]  �          A�33@Z=q�c33A�p�B�(�C���@Z=q��Q�Aw
=Bz=qC���                                    BxW�k�  �          A�\)@0���^�RA���B�  C�,�@0����Aup�B}�\C��                                     BxW�z\  �          A�p�@"�\�aG�A���B��qC���@"�\��\)Au�B~
=C���                                    BxW�  
�          A��H@B�\�mp�A�G�B�k�C��=@B�\��(�Ar{BxC�1�                                    BxW旨  �          A��@>�R�qG�A�(�B��\C�%@>�R��
=As\)Bx�C�޸                                    BxW�N  �          A�p�@!��xQ�A�Q�B��C��\@!����HAs33Bx��C�:�                                    BxW��  �          A�=q@��n{A�B�C��H@���\)Av�\B|ffC�y�                                    BxW�Ú  �          A���?�\)��=qA�ffB�B�C��
?�\)� z�Aq��Bup�C��\                                    BxW��@  
�          A��H?У��R�\A�G�B��C�1�?У���=qAw
=B��3C���                                    BxW���  T          A�\)?���z�HA�
=B��\C�� ?����At(�B{Q�C��\                                    BxW��  T          A���?�
=��ffA�ffB��C��3?�
=�Aj=qBi�C�`                                     BxW��2  T          A���?�{���RA��B�=qC�T{?�{���Ah��Bi�\C�aH                                    BxW��  �          A���?z�H��A��\B��=C�� ?z�H�z�Ad��Bcz�C�                                      BxW�~  
�          A��\@8Q��R�\A�  B�L�C�� @8Q����AtQ�B}��C��                                    BxW�*$  T          A�@1��e�A���B�u�C���@1����
At��Bz��C�XR                                    BxW�8�  T          A��@"�\�xQ�A�{B���C��R@"�\����Ar{Bw��C�+�                                    BxW�Gp  T          A�p�?���z�A�(�B�=qC�j=?���\ApQ�Bs�HC���                                    BxW�V  �          A��H@�R��\)A�\)B��C��@�R���Ao\)Bs�C��=                                    BxW�d�  "          A�\)@#�
�z=qA�{B�\)C��)@#�
���RAq�Bv��C�1�                                    BxW�sb  
�          A��@+��n�RA�z�B�  C��@+���As\)Bx�HC��=                                    BxW�  �          A��@3�
�eA��RB�G�C�@3�
��At(�Bz
=C�Y�                                    BxW琮  �          A��@'��#�
A�ffB�C��@'��љ�A33B�L�C���                                    BxW�T  �          A�\)@Q��aG�A��B���C��@Q���{Ax��B}\)C���                                    BxW��  �          A�\)@   �z=qA�{B�
=C�Q�@   ����Aup�Bw�\C��                                    BxW缠  �          A�\)?���n{A�G�B�G�C�l�?����p�Axz�B|33C��                                    BxW��F  T          A�G�?��H�qG�A�33B�z�C�+�?��H��
=Ax(�B{�C�:�                                    BxW���  �          A�G�?�{�}p�A�
=B���C��?�{����Aw
=By�C�                                    BxW��  �          A���?��R�fffA�33B�\C�AH?��R���Axz�B~z�C��3                                    BxW��8  �          A��H?����^�RA���B�=qC�ff?�����
=AyB�.C��                                    BxW��  �          A�Q�?��j�HA��HB�B�C��\?���z�Aw�B}�
C�W
                                    BxW��  �          A�Q�?��\�mp�A���B�C�p�?��\��Av�HB|��C���                                    BxW�#*  �          A��\?Q���Q�A�(�B�  C�z�?Q��33As�
Bv\)C�޸                                    BxW�1�  �          A�Q�?�p��w�A�ffB��)C��3?�p����\AuBzC�u�                                    BxW�@v  �          A�z�?aG����HA�ffB�
=C��?aG�� ��At��Bx\)C�"�                                    BxW�O  �          A��\?fff�~�RA��RB�ǮC�ff?fff���RAuByC�<)                                    BxW�]�  �          A�z�?&ff��
=A��B�ǮC��?&ff��\AqBsG�C�33                                    BxW�lh  
�          A�
=?5����A���B�W
C�?5�ffAi�BjQ�C�J=                                    BxW�{  �          A���?E���p�A��B���C��?E���Af{Bd�HC�b�                                    BxW艴  �          A�{?G����RA���B�  C�AH?G��z�Af�HBgp�C�w
                                    BxW�Z  �          A�Q�?��R���A�=qB��qC�"�?��R��AiG�Bkz�C�
=                                    BxW�   �          A�(�?��\��  A�p�B�C�Q�?��\��RAm�Bs�C�k�                                    BxW赦  �          A�  ?�ff���A��
B�8RC�ff?�ff�Q�Ah(�BjQ�C�5�                                    BxW��L  �          A��
?���z�A�(�B�ǮC�e?��z�Aj�HBpQ�C���                                    BxW���  �          A�{?���{A���B���C�K�?���G�A��RB�.C�t{                                    BxW��  
�          A��H?��H�'
=A���B��C�g�?��H����A~�HB�B�C��                                    BxW��>  T          A�33?�33�C�
A�z�B��3C�Ff?�33��RA|Q�B�{C�|)                                    BxW���  �          A�
=?����G
=A�ffB��)C���?�����Q�A{�
B��HC��\                                    BxW��  �          A�{?���AG�A��B�W
C���?����
=A~=qB�z�C��                                    BxW�0  �          A�=q?�p��C33A�p�B��C���?�p���  A}�B��3C��3                                    BxW�*�  �          A��?�p��6ffA��HB�
=C�^�?�p��ᙚA}��B�p�C�ٚ                                    BxW�9|  �          A�{@�\�3�
A�G�B�#�C��q@�\���A~�\B���C�R                                    BxW�H"  �          A�
=@&ff�.{A��
B�\)C��@&ff��ffA�
B�33C�C�                                    BxW�V�  �          A��@7
=�%A��B�C��@7
=�ڏ\A�{B�\)C�Z�                                    BxW�en  T          A���@Vff�   A��RB�� C��f@Vff��
=A~�\B�u�C�AH                                    BxW�t  �          A�z�@s�
�Q�A�  B�.C�j=@s�
��33A~�\B��C��                                     BxW邺  �          A���@j=q���A�Q�B���C�.@j=q�ϮA~�RB��
C���                                    BxW�`  �          A���@$z��5A�G�B�p�C��@$z�����Az=qB��
C�                                    BxW�  �          A�z�@!G��AG�A��HB��RC��{@!G���{Ax��B�C���                                    BxW鮬  �          A�p�?fff�eA�(�B�{C��?fff��\)At��B|33C�P�                                    BxW�R  �          A��?W
=�R�\A�ffB��C�.?W
=��ffAv�\B�{C�9�                                    BxW���  	          A���?}p��b�\A�B�33C���?}p���AtQ�B|z�C���                                    BxW�ڞ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��D              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�6              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�#�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�2�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�A(              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�O�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�^t              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�m              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�{�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�f              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW꧲              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�X              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�Ӥ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��J              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�<              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�+�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�:.              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�H�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�Wz              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�f               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�t�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�l              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW렸              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�^              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�̪              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��P              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�B              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�$�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�34              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�A�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�P�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�_&              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�m�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�|r              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW왾              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�d              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�
              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�Ű              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��V              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW� H              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�,:              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�:�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�I�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�X,              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�f�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�ux              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��j              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��\              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��N              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�%@              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�3�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�B�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�Q2              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�_�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�n~              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�}$              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�p              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�            A��
?�ff��RAfffBm��C�` ?�ff�A�A8Q�B.  C��                                    BxW  	�          A��
?�  �(�A^=qB`�\C�
?�  �K�
A,��B �C�@                                     BxW��b  
(          A�>��
�{AZ{BZ�C���>��
�PQ�A&�HBC���                                    BxW��  �          A�{?���
{Ae��Bk�RC���?���D(�A6ffB+��C�c�                                    BxW��  �          A��>\)�Ag�Bo��C�~�>\)�@��A9p�B/�RC�XR                                    BxW��T  �          A�\)>�  ��p�AlQ�Bz33C��>�  �7\)AA�B9�C��                                     BxW� �             A��?�ff�\)Ac�Bi\)C�s3?�ff�D��A4  B)\)C�q�                                    BxW��  D          A���>�����
A`��Bep�C���>����H��A0(�B%(�C���                                    BxW�F  T          A�p�?=p��ffAaG�Bfz�C�c�?=p��G\)A1�B&G�C���                                    BxW�,�  �          A����{���Aj�\Bw�C�����{�9�A>�\B6��C�&f                                    BxW�;�  T          A�p�>L���ffAh��Bs
=C���>L���=A;�B2�C�xR                                    BxW�J8  "          A��?����Ae�Blp�C��H?��C
=A6{B,{C�:�                                    BxW�X�  
�          A��>u�
ffAd(�Bj�C��=>u�DQ�A4��B*�C��                                    BxW�g�  �          A�G�?0����Ab�\Bi(�C�AH?0���D��A2�HB(�
C���                                    BxW�v*  T          A���?s33�33Ac�Bi��C��?s33�D��A4  B)\)C�33                                    BxW��  T          A��?n{�\)Ac
=BiG�C��?n{�D��A3�B)
=C�(�                                    BxW�v  "          A�G�?����G�AaG�Bf��C��\?����F=qA1�B&��C���                                    BxW�  "          A��?�p��Q�A_
=Bcp�C��H?�p��H��A.=qB#Q�C���                                    BxW��  T          A�\)?}p���Ag\)Bp�C�n?}p��>�RA9��B0ffC�`                                     BxW�h  "          A�33?\�33Ad(�Bk�
C�)?\�AG�A5��B+��C��
                                    BxW��  
�          A���@����A[�B^�C�:�@��H(�A*�\B��C�H�                                    BxW�ܴ  �          A���@
�H�(�AW\)BX=qC�p�@
�H�N{A$z�B��C��=                                    BxW��Z  
�          A�=q?J=q�\)Ab�HBlQ�C��?J=q�A�A4Q�B+�HC��H                                    BxW��   �          A��
��ff��Ah��BxffC�P���ff�6�HA=�B7��C��q                                    BxW��  �          A���B�\��33Ai�ByQ�C�"��B�\�5A>{B8�HC�{                                    BxW�L  "          A�녾�����{Ai�Bx�C��H�����733A=��B7�HC���                                    BxW�%�  �          A��
>�ff���RAf�\Bt�C���>�ff�:�\A9�B3z�C�R                                    BxW�4�  �          A�  =��
��  Ah��Bw��C�J==��
�8  A=�B7{C�1�                                    BxW�C>  
�          A�녾.{��G�Ap(�B�L�C�L;.{�+33AHQ�BE�HC��                                    BxW�Q�  �          A�녽�Q��ᙚAnffB�\)C��H��Q��.�RAE��BB  C�                                    BxW�`�  �          A�{����   Af�HBs�C�f����;\)A9�B3
=C��f                                    BxW�o0  �          A�  �z����
AiBy�C��ÿz��6ffA>�\B8�HC���                                    BxW�}�  T          A��
���H��\AiBz{C�(����H�5A>�RB9ffC��                                    BxW��|  T          A��Ǯ���
Aip�By�C����Ǯ�6=qA>{B8��C�f                                    BxW�"  T          A��
����z�Ag33Bu33C�{���9A:�HB4�C���                                    BxW��  �          A���(��=qAd��BqG�C�xR��(��<��A7�B0�\C���                                    BxW�n  �          A����33�\)Ac�
Bp  C��f��33�=A6{B/=qC�&f                                    BxW��  �          A���
=q�(�A`��Bk  C�.�
=q�A��A2{B*Q�C��R                                    BxW�պ  "          A���E��p�Ad��Bq��C�K��E��<(�A7�B1
=C�"�                                    BxW��`  T          A����33��\)AeG�Br��C��f��33�:�\A8��B2G�C�/\                                    BxW��  �          A���Q�� ��Ae�Bq��C�R�Q��;�
A8  B1ffC�                                      BxW��  �          A����J=q� Q�AeG�Br��C�,ͿJ=q�;33A8z�B2  C�\                                    BxW�R  �          A�����Ae�BqC�Y�����<��A7�
B1  C��q                                    BxW��  �          A��
��33��Ad(�Bo�C�����33�>{A6ffB/(�C�(�                                    BxW�-�  �          A�녽��
=Ad��BpC��R���=A733B/��C��R                                    BxW�<D  T          A��
<��
��AeBr��C�{<��
�<(�A8��B1�
C�                                    BxW�J�  �          A��>#�
��G�Ah  Bw  C��R>#�
�8Q�A<  B633C�ff                                    BxW�Y�  �          A��>�  ��Aj{Bz
=C��>�  �6=qA>�RB9=qC���                                    BxW�h6  "          A�  ��\)��ffAm��B�8RC��{��\)�0��AD  B?��C��                                    BxW�v�  �          A�  =�G���{Amp�B�L�C�p�=�G��0��AD  B?��C�H�                                    BxW�  "          A�=q�#�
��G�Ar�HB�ffC��ü#�
�((�AL  BJ  C���                                    BxW�(  T          A�=q��������Ar{B��=C��������)��AJ�RBHQ�C���                                    BxW��             A�ff>B�\�ϮAs�B���C��)>B�\�'�AL��BJ�
C���                                    BxW�t  �          A���=��
�љ�As�B�k�C�Y�=��
�(��AL��BJ{C�7
                                    BxW��  T          A�33?(���z�Au��B���C��
?(��&�RAO\)BL��C���                                    BxW���  T          A�33>�ff�أ�As
=B��)C��>�ff�+�AK33BG
=C�0�                                    BxW��f  �          A��?:�H��Am��B}G�C�Ф?:�H�4Q�AC
=B<C���                                    BxW��  �          A�33?   ���
Aj�\Bv�
C���?   �:=qA>{B633C�=q                                    BxW���  �          A�G�?!G����Af�RBo��C�'�?!G��@  A8��B/\)C�~�                                    BxW�	X  
�          A�G�?:�H�Q�Af�HBpQ�C���?:�H�?�A8��B/C��q                                    BxW��  
�          A�G�>Ǯ��z�Al��Bz��C�t{>Ǯ�7\)AA�B9��C��R                                    BxW�&�  2          A�33?5���Aip�Bt��C��=?5�;�
A<��B4G�C���                                    BxW�5J  �          A�\)>����Ag33Bpz�C��{>��?�A9�B/�HC�
                                    BxW�C�  
�          A�33=��
��Ag�Bq��C�C�=��
�>�HA9B0��C�.                                    BxW�R�  T          A���>\��Aip�Bu��C�^�>\�:�RA<��B533C��                                    BxW�a<  �          A�=q>������Aw\)B��C�J=>���AS�
BUQ�C�                                    BxW�o�  �          A�{>.{��HA�ffB�\C�\>.{�ۅAo�
B���C���                                    BxW�~�  �          A�ff>W
=�N�RA��B�aHC��f>W
=��33AjffBz=qC��\                                    BxW�.  "          A�z�?G���
=A|Q�B��{C�w
?G���\A\(�Ba
=C�p�                                    BxW��  "          A���?����\Ayp�B�k�C���?��33AV�RBXp�C���                                    BxW�z  �          A�z�=u��=qA|  B�Q�C�T{=u�  A[�B`�C�.                                    BxW�   �          A���>B�\����A}p�B��\C�#�>B�\�A]��Bb��C��q                                    BxW���  �          A��H>8Q���\)Az�RB�k�C���>8Q��AXz�BZffC��f                                    BxW��l  �          A��
<��  Aa��Bn��C�q<��=p�A4  B.(�C�{                                    BxW��  �          A�p��8Q���\AQ�BS  C���8Q��RffA��B�C�p�                                    BxW��  �          A����z��Q�AV�\BZz�C�!H��z��M�A#�
B{C�Z�                                    BxW�^  �          A���:�H� Q�APz�BQp�C��Ϳ:�H�S�
A�
B33C�l�                                    BxW�  
�          A�������
=AS�
BV�C��Ὲ���O�
A ��Bp�C��f                                    BxW��  �          A����G��  AX  B]�C��׿�G��J{A&�\B��C��R                                    BxW�.P  �          A�  ��\� ��Af=qBr��C�.��\�;�A9p�B2�\C��                                     BxW�<�  �          A���W
=����Ag33Bv�\C�9��W
=�7�
A;�B6Q�C�y�                                    BxW�K�  �          A��\>�{��z�Am�B}�HC�L�>�{�333AC
=B=�C���                                    BxW�ZB  �          A��\@��s33A\)B�Q�C���@���Ad��BnC��                                    BxW�h�  �          A�z�@,������Az�HB��C�XR@,���\)A\��Bb33C���                                    BxW�w�  T          A�=q@(����RAv�RB�8RC���@(��(�AT��BW  C�|)                                    BxW�4  "          A�=q?�(���Q�AzffB��C�+�?�(��=qAZ�\B_�\C���                                    BxW��  �          A�=q<#�
���A|  B��)C�<#�
�=qA\(�Ba��C��                                    