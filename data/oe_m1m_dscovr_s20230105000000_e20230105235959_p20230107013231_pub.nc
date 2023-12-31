CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230105000000_e20230105235959_p20230107013231_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-01-07T01:32:31.160Z   date_calibration_data_updated         2022-11-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-01-05T00:00:00.000Z   time_coverage_end         2023-01-05T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        }   records_fill        #   records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxW�  �          A�(��z���  AxQ�B�C��R�z���AVffBY��C�AH                                    BxW�&  �          A�(��=p���z�Ao�
B�\)C��=�=p��,(�AH(�BD�C�                                    BxW���  �          A��R��33��z�AlQ�B
=C�p���33�/
=AC�B?�
C�XR                                    BxW��r  �          A�33���H���AZ�\B\�RC�����H�K33A)�BC���                                    BxW��  �          A�p���=q��z�Az{B�B�C�����=q��AW�BX��C�8R                                    BxW��  �          A�\)���R���RAzffB��)C�q쾞�R���AW�BW�HC�R                                    BxW��d  �          A��>������RA~�RB�Q�C��f>����ffA_
=Bb�HC��                                    BxW�

  
�          A���    ����Ap��B��3C��q    �,Q�AI�BE��C���                                    BxW��  �          A���L����{As\)B�� C�'��L���)AL��BIG�C�xR                                    BxW�'V  �          A�G�>\)��=qAs
=B���C���>\)�+�AK�
BG�C�^�                                    BxW�5�  �          A�33������
Aj=qBv��C�K�����9p�A>�RB733C��
                                    BxW�D�  T          A���G����HAk�BzffC�{�G��5p�AAG�B:��C�
=                                    BxW�SH  �          A�ff��\��G�Ay�B��HC��{��\�$��AT  BPG�C���                                    BxW�a�  �          A��׽�G���Q�Ax(�B�u�C����G��((�AR=qBM\)C��3                                    BxW�p�  T          A�ff>�
=�ƸRAyp�B�z�C��{>�
=�#�
AT��BQ�C�/\                                    BxW�:  �          A��\<��
�߮At��B��
C��<��
�.�\AL��BF=qC�                                    BxW��  �          A��\�\)��ffAq�B~�RC�xR�\)�4��AG\)B?Q�C��f                                    BxW���  T          A��Ϳ�ff��Ae�Bj(�C��{��ff�E�A7
=B+{C���                                    BxW��,  T          A����k�����Ak
=BwC�� �k��7�
A@(�B8C���                                    BxW���  �          A�������  An=qB|��C�,;��4z�ADz�B=��C���                                    BxW��x  T          A�p��B�\�z�Ad��BlC�Z�B�\�A��A7
=B-�C���                                    BxW��  �          A�
=�s33���AjffBw{C����s33�8  A?�B833C���                                    BxW���  "          A�\)����AaBgQ�C�=q����EA2�\B(=qC��)                                    BxW��j  T          A�=q���
��Q�Aw�B�L�C��q���
�'�AQBM�\C�#�                                    BxW�  �          A��H����{AuB�\)C�w
���-��ANffBG�C���                                    BxW��  "          A��;����RAs�
B�\)C�g�����1�AK\)BCC��)                                    BxW� \  �          A�=q���
��ffApz�B~z�C������
�4  AG33B?�\C�˅                                    BxW�/  �          A�(������\)Am�Bz33C�o\����7�AC�B;Q�C���                                    BxW�=�  T          A���!G���A^ffBb�C�׿!G��H��A.�\B#��C��                                    BxW�LN  �          A�33�5�33Ad��BmQ�C��)�5�@  A7�B.�RC�P�                                    BxW�Z�  |          A��H=��
�\)Ak�BsffC�H�=��
�>=qA?\)B4�RC�1�                                    BxW�i�  �          A��
>k���Ar�HB}p�C�ٚ>k��6�RAH��B>�
C���                                    BxW�x@  T          A��
>���\)Am��Bt(�C��>���>�\AAp�B5��C�"�                                    BxW���  �          A��>��
��HAk�
Bp�
C��>��
�Ap�A>�RB2\)C��                                    BxW���  �          A�?0���{Ak�
Bq=qC�W
?0���@��A?
=B2�C��H                                    BxW��2  �          A�{?=p���
Ak\)Bo��C�|)?=p��B=qA>=qB1ffC��q                                    BxW���  
�          A�(�?&ff�	p�Aj�RBn=qC�*=?&ff�C�A=�B0
=C��f                                    BxW��~  �          A�  ?+����Aj�RBn��C�@ ?+��C
=A=p�B0p�C���                                    BxW��$  �          A�=q?=p���An=qBt
=C���?=p��>�\AB=qB6  C��f                                    BxW���  �          A�Q�?J=q� Q�Ap(�Bw  C�˅?J=q�<  AD��B9
=C���                                    BxW��p  �          A�(�?(����RAn�RBt�HC�K�?(���=AC
=B6�C��{                                    BxW��  �          A��?8Q���
Amp�Bsz�C�� ?8Q��>�\AA��B5��C���                                    BxW�
�  
�          A�(�?}p��Q�Amp�Br�RC�h�?}p��?
=AAp�B5
=C�^�                                    BxW�b  "          A�=q?��� ��Ao�Bv{C��?���<  ADz�B8��C��                                    BxW�(  �          A�Q�?������Ap  Bv�HC���?����:{AE��B9�HC�ٚ                                    BxW�6�  �          A�ff?�  ��An�\Bt�C�:�?�  �<��AC\)B7{C���                                    BxW�ET  �          A�Q�?�G���Amp�BrQ�C�5�?�G��>{AAB5Q�C��H                                    BxW�S�  "          A�ff?�ff� ��Ao33Bu=qC�xR?�ff�;�ADz�B8\)C��f                                    BxW�b�  T          A�ff?���{An�RBtQ�C��R?���<��AC�B7\)C�Z�                                    BxW�qF  �          A�z�?�33���
Ar�RB{�C�)?�33�5�AIB>��C�!H                                    BxW��  �          A�z�?�Q���p�Ap��Bw\)C�&f?�Q��9�AF�\B:��C���                                    BxW���  �          A�(�?�����Ap��Bxp�C��{?���8(�AG
=B;�HC��\                                    BxW��8  "          A�  ?�
=���
Aq��Bz��C�5�?�
=�5p�AH��B>Q�C�5�                                    BxW���  �          A�Q�?޸R���Aqp�By�C�U�?޸R�7
=AH(�B<�C�Q�                                    BxW���  
�          A�?����   An{Bu�C���?����:=qAD  B8��C�ٚ                                    BxW��*  
�          A��?����G�ApQ�Bx�C���?���7�AG
=B<G�C���                                    BxW���  �          A��
?��
��=qApQ�BxC��?��
�7�
AF�HB<\)C�,�                                    BxW��v            A�  ?��
�Al(�Bq33C���?��
�?
=A@��B4��C�u�                                    BxW��  
�          A�{?��
���Ao\)Bv\)C���?��
�:{AEp�B:{C�"�                                    BxW��  �          A�ff?�\)���\AqG�Bx��C���?�\)�8(�AH  B<�C�g�                                    BxW�h  
Z          A�ff?�����RApQ�Bw{C��
?���9AF�RB:��C�(�                                    BxW�!  �          A��H?�Q��	�Ak�Bm�C��?�Q��B�\A?\)B1�C���                                    BxW�/�  �          A���?�ff�(�Al��Bo33C�T{?�ff�A�A@��B333C��                                    BxW�>Z  T          A��?�����\)Aq�BwffC��3?����:=qAHQ�B;�\C�<)                                    BxW�M   �          A�
=?��� Q�AqG�Bv��C��\?���:�RAG�B;  C�&f                                    BxW�[�  �          A��?�����Aq��Bw(�C��R?���:=qAH  B;z�C�+�                                    BxW�jL  
(          A�G�?��
���RAtz�B{��C���?��
�6�\AL  B@
=C�33                                    BxW�x�  
�          A��?�=q��HAp  BtG�C��q?�=q�<��AE�B8�RC�8R                                    BxW���  �          A�G�?p����An=qBp�
C�*=?p���@��AC
=B5{C�:�                                    BxW��>  "          A�\)>��
�	��AmBo�C��>��
�BffAB{B3��C��H                                    BxW���  �          A�>�{�
{An=qBo�C�#�>�{�B�HAB�\B3C��\                                    BxW���  �          A��>�z����Am�Bm{C��>�z��EG�A@��B1\)C��=                                    BxW��0  C          A��
>��R�G�Aj=qBh��C��q>��R�H��A<��B-
=C��R                                    BxW���  
�          A�=q?��\)AiBf��C�Ǯ?��JffA<(�B+z�C�K�                                    BxW��|  �          A�Q�?���
Ai��Bfp�C��)?��J�HA;�
B+
=C�,�                                    BxW��"  
�          A�  ?+���ffAt��ByC�e?+��9��AK�
B>�C��                                    BxW���  T          A�=q?��
��\Aw�
B~�C���?��
�4z�APz�BD  C���                                    BxW�n  �          A���?aG���An{Bl�C���?aG��EG�AB=qB1��C��                                    BxW�  �          A��H?(�����Ag
=BaQ�C���?(���O\)A8Q�B&\)C�s3                                    BxW�(�  �          A���?#�
�G�Ak�Bi(�C�H?#�
�Hz�A>�HB.=qC�t{                                    BxW�7`  T          A���?u� z�AbffBZ=qC���?u�T��A2=qB�C�3                                    BxW�F  
�          A�\)?�ff�%�A_�
BU�
C��?�ff�Xz�A.�RBG�C�7
                                    BxW�T�  �          A�
=>��R�ffAiBd�
C��{>��R�L��A<  B*(�C��{                                    BxW�cR  "          A�
=�B�\��
An{Bkp�C�g��B�\�G\)AA�B0��C���                                    BxW�q�  T          A�\)=�G����Ak�BfC�W
=�G��K�A>=qB,=qC�@                                     BxW���  "          A�p�?
=���Am��BiC��
?
=�H��AA�B/p�C�U�                                    BxW��D  T          A�\)>����Ap(�Bn(�C�Q�>���E�AD��B3�HC���                                    BxW���  
�          A��?��
�z�Aj=qBc�C�{?��
�N�\A<Q�B)(�C�G�                                    BxW���  �          A�{?�z���Am�Bg�C�T{?�z��JffA@z�B-z�C�+�                                    BxW��6  �          A�  ?Y���G�An�HBj\)C���?Y���Hz�AB�HB0�C��\                                    BxW���  
Z          A��?!G���Am�BhG�C���?!G��J{A@��B.p�C�n                                    BxW�؂  �          A�  ��p���Ak\)Bd��C�޸��p��Mp�A>{B+  C�+�                                    BxW��(  "          A�z�
=q�(�An�\Bhz�C�S3�
=q�J�HAB{B.�
C��f                                    BxW���  T          A���=�Q����Aq�Bl
=C�J==�Q��H(�AEp�B2ffC�5�                                    BxW�t  
�          A�z�?����HAo�BiC��{?���IAC�B0Q�C�>�                                    BxW�  !          A��R?fff�z�As\)Boz�C��\?fff�DQ�AH��B6G�C��                                    BxW�!�  T          A���?z����Ao33Bh=qC��=?z��K\)AB�HB/  C�O\                                    BxW�0f  �          A�
=?E��ffAp��Bj\)C�c�?E��IG�AE�B1Q�C��q                                    BxW�?  
�          A�G�?�z��(�Au�BrQ�C��H?�z��@��AL��B9��C��=                                    BxW�M�  
�          A�\)?�33��Aw�
Bu(�C���?�33�>{AO�B<��C��3                                    BxW�\X  "          A��?�=q�p�Ax(�Bu33C�Y�?�=q�>ffAO�B=  C��f                                    BxW�j�  �          A��?����33As�BmffC���?����FffAH��B4��C���                                    BxW�y�  "          A��?�ff�Q�An�RBe33C�(�?�ff�N{AB=qB,��C�W
                                    BxW��J  "          A��?
=q�&�RAdz�BW\)C�xR?
=q�Yp�A4��B�HC�!H                                    BxW���  T          A���>k��#�Af�RBZ�C���>k��W
=A7�
B"�C�}q                                    BxW���  
�          A����33�$Q�Ae�BY��C����33�W33A733B!�C�B�                                    BxW��<  �          A��@33��HAw�Br��C���@33�>�HAO�B;�HC��q                                    BxW���  �          A�  ?У��	�Av�RBqz�C�Z�?У��A��AM�B:{C���                                    BxW�ш  �          A�{?�G����Aw�Brp�C��{?�G��@z�AO33B;G�C�+�                                    BxW��.  
�          A�ff?��	�Aw\)Bq{C�J=?��A��AN�RB:(�C���                                    BxW���  T          A�z�?�\)���AuBn�\C��q?�\)�D  AL��B7�C�U�                                    BxW��z  
�          A��\?���{Aup�Bm�C��q?���D��AL  B6C�Y�                                    BxW�   T          A��\?�����Atz�Bl
=C�  ?����E�AJ�HB5p�C�y�                                    BxW��  �          A��H?��z�At��Bk�RC���?��F�HAK33B5(�C�7
                                    BxW�)l  
Z          A��?�\)�=qAt(�BjG�C�˅?�\)�HQ�AJ=qB3��C�B�                                    BxW�8  
�          A�\)@ ���(�As\)BhQ�C�#�@ ���IAH��B2{C���                                    BxW�F�  "          A�33?�p���RAqG�BeC���?�p��K�
AF�\B/��C�l�                                    BxW�U^  �          A��?�z��Q�AmB`�RC���?�z��PQ�AA�B*�\C�(�                                    BxW�d  "          A�G�@33��\Ao
=Bb33C���@33�N�RAC�B,G�C���                                    BxW�r�  "          A�G�?��R��\Au�Bk=qC���?��R�HQ�AK�B5{C�g�                                    BxW��P  "          A�33?���(�AxQ�Bp�C�  ?���B�HAP(�B:�RC��q                                    BxW���  �          A���?����\)AvffBnQ�C��\?����EG�AMB8=qC��f                                    BxW���  �          A��R?z�H�z�Aw�
Bq33C�1�?z�H�B�RAO�
B;�C�O\                                    BxW��B  f          A�33?����\)At��Bk(�C�N?����H��AK�B5=qC�o\                                    BxW���  �          A��?�ff�33Aw
=Bn��C�  ?�ff�D��AN�RB8��C��                                    BxW�ʎ  �          A�G�?�ff���Az�HBt\)C�T{?�ff�?�AT  B>�
C��                                    BxW��4  T          A�p�?���AvffBl�C�@ ?���G
=AMB7\)C�b�                                    BxW���  T          A�G�?�
=�{Av�RBnp�C�\)?�
=�C�AN�HB9p�C��=                                    BxW���  �          A�33@{�Q�Aw
=Bn�C�)@{�AAO�
B:(�C�33                                    BxW�&  �          A���@�����RA}��By��C��R@���6ffAYG�BE�HC�t{                                    BxW��  �          A�33?�����A��\Bz�C��?��2=qA^{BKG�C��                                    BxW�"r  �          A���?�\)��p�A�p�B��RC�%?�\)�/
=A`��BO�C�33                                    BxW�1  �          A�33?���(�Ax(�Bp(�C�Ǯ?���AG�AQp�B;�HC�7
                                    BxW�?�  T          A�\)?��H���Az�\Bs�C���?��H�>ffAT��B?�C�R                                    BxW�Nd  T          A�?���� ��A�(�B|  C��q?����7�
A\(�BG�RC�k�                                    BxW�]
  �          A�Q�?}p���A�(�Bz�RC�n?}p��:ffA[�BFQ�C�l�                                    BxW�k�  T          A�p�?Y����  A�33B�p�C�!H?Y���3�A_�BL�\C�+�                                    BxW�zV  "          A�ff?��  A�=qBz�C��?��:�\A[�
BF�C��                                     BxW���  T          A�{?�G��33A�  Bz�C���?�G��9A[�BF�RC�}q                                    BxW���  �          A�Q�?O\)�G�A�Byz�C�˅?O\)�;�A[33BE�\C��q                                    BxW��H  T          A���?�����A}��Bu33C��)?���>�\AXQ�BA�RC�T{                                    BxW���  �          A���@G����AzffBo�C�:�@G��AG�ATz�B<�HC�P�                                    BxW�Ô  �          A���?�����Ax��Bl��C��?����E�AQB9C�~�                                    BxW��:  T          A��\?:�H��RA{33BqQ�C�W
?:�H�C33AU�B=�HC��
                                    BxW���  "          A�
=?��H�G�Aup�BgG�C���?��H�L  AM�B4(�C��
                                    BxW��  T          A��?�  �  At  Bd�RC���?�  �N=qAK33B1C��f                                    BxW��,  T          A�G�?�����As�
Bd(�C�/\?����O33AK
=B1=qC�k�                                    BxW��  f          A�=q?���)Alz�BX
=C�aH?���YAAG�B%z�C�l�                                    BxW�x  �          A�Q�?���%��Ap(�B\�C���?���V�\AEB*�C��)                                    BxW�*  �          A��?�
=��HAs\)Bb��C�ff?�
=�Pz�AJ�\B033C��R                                    BxW�8�            A�z�?�G��333Ae�BN�HC�xR?�G��aG�A9�BC���                                    BxW�Gj  �          A��?��\��A��B}33C�|)?��\�6�RAap�BK33C�.                                    BxW�V  
�          A�33?�ff��HA}�Bq�HC�\)?�ff�B�RAX��B?��C�xR                                    BxW�d�  �          A�G�?h���%�A�Bi�C���?h���\��Ae�B7�C���                                    BxW�s\  �          A��?fff�)A���Bf�
C�l�?fff�`(�Ab�\B4��C���                                    BxW��  T          A��?^�R�.=qA��Bc�C�E?^�R�d  A`  B1C��q                                    BxW���  �          A�  ?Y���.�RA��
Bc�C�8R?Y���dQ�A_�B1z�C��3                                    BxW��N  
�          A��?�
=�0(�A��HBa=qC��?�
=�e�A]B/�
C�]q                                    BxW���  �          A���>����"�HA���Bl�C���>����Y��Ahz�B;z�C���                                    BxW���  
�          A�p�=u��HA�=qBp�C�*==u�V=qAlQ�B?33C�                                      BxW��@  T          A��<��33A��Br=qC��<��R=qAk�BA{C�\                                    BxW���  T          A�{>�Q��   A�=qBn=qC�
=>�Q��V=qAhQ�B=G�C��f                                    BxW��  �          A�z�?p���.�RA��Ba�\C�w
?p���b�\A\��B0�
C��f                                    BxW��2  �          A�G�?\�?
=A}BS33C���?\�pQ�AO�B"��C��                                    BxW��  T          A�  ?�33�AG�A}BQ�
C��H?�33�r=qAO\)B!��C��                                    BxW�~  �          A���?���8(�A�BY��C���?���j�\AW33B)�C��q                                    BxW�#$  �          A���?���9�A��RBXp�C�0�?���j�HAU�B(ffC���                                    BxW�1�  �          A���?�  �:ffA�  BV�HC���?�  �k�AS�B'{C���                                    BxW�@p  �          A��H?�=q�9G�A�z�BW�
C��?�=q�j�\AT��B(=qC�3                                    BxW�O  "          A��R?�G��9��A�  BV�C�S3?�G��j�RAT  B'�\C�n                                    BxW�]�  T          A�
=?��5A
=BY33C���?��f�\AT  B)�
C��                                    BxW�lb  �          A���?�\)�733A}��BW�
C�h�?�\)�g\)ARffB(��C��{                                    BxW�{  �          A��R?�G��3�
A�  BZ��C�1�?�G��dz�AU��B+�
C��                                    BxW���  �          A�?�=q�C�AvffBM=qC���?�=q�q�AI�B\)C���                                    BxW��T  �          A���?�33�6ffA~=qBX�
C��?�33�fffAS�
B)��C�H�                                    BxW���  �          A�
=?J=q�-G�A��RBa�C�R?J=q�^�HA\��B2C��H                                    BxW���  �          A�=q?���6�HA|��BX
=C���?���f=qAR�\B)z�C�+�                                    BxW��F  �          A���?�ff�G\)Aq��BI=qC��R?�ff�t(�ADQ�B�HC�n                                    BxW���  T          A���?��
�M�Al(�BCG�C�e?��
�xz�A>{B(�C��                                    BxW��  �          A��?���P(�Aj{B@�\C�` ?���z�HA;�B�\C��                                    BxW��8  �          A�G�?��H�S\)Ag\)B=ffC��?��H�}p�A8z�B��C�3                                    BxW���  �          A�33?����UG�Ad��B;
=C���?����~�RA5�Bp�C�C�                                    BxW��  �          A�@
=�_�A[�
B0��C�L�@
=��p�A+
=BQ�C��=                                    BxW�*  �          A�  ?���=��Aw�BQ�C��H?���j�HAM�B$33C�9�                                    BxW�*�  T          A������!��A��RBl�C�AH����S�AhQ�B>��C�n                                    BxW�9v  �          A��þ.{�&�RA���Bg�RC����.{�W�Ad  B:ffC��f                                    BxW�H  
�          A�
=���(z�A�z�Bf=qC��f���X��Ab�HB9{C���                                    BxW�V�  T          A��H=�\)�7�
A~=qBX�\C�/\=�\)�e�AU��B+�\C�&f                                    BxW�eh  �          A���=�G��0��A��
B_�C�Ff=�G��_�A\Q�B2G�C�7
                                    BxW�t  
�          A��H<#�
�-�A��HBb{C��<#�
�\z�A_33B5ffC�f                                    BxW���  �          A�
=    �.�\A��\B`�C�      �]A^�\B4ffC�                                      BxW��Z  T          A�
=>���/�A�=qB`  C�\>���^ffA]B3�C���                                    BxW��   T          A�G�?+��1p�A�B^ffC���?+��_�
A\��B2G�C�^�                                    BxW���  �          A�G�?��9p�A}BW33C�ff?��f�\AUB+=qC�                                      BxW��L  
�          A�\)?���C�AvffBN�C�H�?���n�HAL��B"Q�C��                                    BxW���  �          A�
=>L���A�Aw
=BOp�C�w
>L���mG�AM�B#C�aH                                    BxW�ژ  �          A�33?+��F�RAs�BK
=C���?+��q�AIB�\C�C�                                    BxW��>  �          A��?aG��J�\Ao�BF�
C���?aG��t  AE�B��C���                                    BxW���  T          A��H?:�H�G�AqBI��C���?:�H�qG�AH  B�C�aH                                    BxW��  �          A�z�?��\�I��An�RBF�C�P�?��\�r�RAD��B  C���                                    BxW�0  "          A�=q?�=q�P��Ah  B?�C�^�?�=q�xz�A=�B��C���                                    BxW�#�  �          A��?�Q��[
=A[�
B4
=C�y�?�Q���  A/�B	z�C��                                    BxW�2|  �          A�?�  �_33AW\)B/\)C���?�  ����A*ffB  C�7
                                    BxW�A"  �          A�  ?z�H�W
=Af�RB;C�
?z�H�}A;33Bz�C��                                    BxW�O�  "          A�ff?�=q�[�Ac�
B7�HC�=q?�=q����A7�
B��C���                                    BxW�^n  �          A���?�G��S
=Alz�B@�C�/\?�G��z�\AB{BC��
                                    BxW�m  �          A�z�?����T��Ai�B>z�C�]q?����{�A?�BC�                                      BxW�{�  
�          A�z�?k��YAe�B:  C��?k���A:�RBffC��                                    BxW��`  T          A�=q?k��[\)Ac�
B8{C��?k���ffA8��B��C��f                                    BxW��  
�          A�{?u�YG�Ae�B9C��?u�~�HA:ffB�C��R                                    BxW���  T          A��?=p��Y�Ac\)B8��C���?=p��
=A8��Bz�C�Q�                                    BxW��R  �          A��
?   �[
=A]��B5=qC�?   �
=A3
=B=qC��                                    BxW���  T          A��>�(��\z�A[\)B3\)C��f>�(���
A0��B
�\C��f                                    BxW�Ӟ  
�          A�{?�z��b�\AU�B-{C�U�?�z���ffA*�RB�\C��                                    BxW��D  T          A���?���b�RATQ�B,(�C�4{?����Q�A)G�BC��                                    BxW���  "          A�(�?���ap�A\��B1  C��
?����Q�A1B�
C�Y�                                    BxW���  
�          A�Q�?��H�^{A`��B4��C�}q?��H���HA6�RB��C�&f                                    BxX 6  �          A�  ?�ff�`��AW�
B/(�C�"�?�ff��\)A-BQ�C���                                    BxX �  
�          A��
?���b{AZ�RB/�HC�R?����=qA0Q�B(�C���                                    BxX +�  "          A�?\�k�AV=qB(z�C��{?\��z�A*�\B
=C��
                                    BxX :(  �          A�(�?��
�g�
A[\)B-{C�f?��
����A0z�B��C���                                    BxX H�  �          A�=q?�{�i�AZ�\B+�C�&f?�{��p�A/�B�HC��H                                    BxX Wt  T          A���?���m��AV=qB'\)C�o\?����G�A*�HB �C�                                    BxX f  "          A���@G��o�ATQ�B%�C��R@G���{A(��A�
=C�c�                                    BxX t�  �          A�
=@{�rffAPz�B!G�C���@{��
=A$��A�  C��                                    BxX �f  �          A��R@�r�RAO33B ��C�b�@��
=A#�A��HC��                                    BxX �  T          A�Q�@���rffANffB 33C�� @�����RA#
=A�z�C���                                    BxX ��  
�          A�(�@�r�RAMB�
C�c�@���RA"�RA�{C��                                    BxX �X  T          A��
@���r�\ALz�B(�C�y�@����z�A!��A��C��)                                    BxX ��  T          A��@z��s33AK
=B�C�XR@z�����A Q�A�G�C��                                     BxX ̤  �          A��@!G��r�RAK33B33C���@!G���Q�A ��A�  C�33                                    BxX �J  �          A�\)@G��u�AHz�B�C�5�@G���G�A�A홚C���                                    BxX ��  "          A��R@  �p  AL��B �\C�C�@  ��
=A#
=A�G�C���                                    BxX ��  �          A���@(��p(�AL��B ��C�&f@(���
=A#\)A��C���                                    BxX<  �          A�(�?�Q��j�\AQB&=qC��?�Q�����A)p�B�\C�U�                                    BxX�  �          A��?��j�\AP(�B%ffC��)?���ffA(  B �HC�O\                                    BxX$�  �          A��?���d  AP(�B(�C��{?�����A)G�B\)C�*=                                    BxX3.  �          A���?ٙ��c�AO�
B(�HC�h�?ٙ����RA)G�B�C��                                    BxXA�  �          A�Q�?��H�c
=AO�B(�HC�p�?��H��ffA)�B�HC��                                    BxXPz  �          A�?����c
=AN{B(Q�C��=?�����Q�A(  Bp�C��{                                    BxX_   �          A�p�?����f{AI�B$�C��)?�����\)A#�B ��C�n                                    BxXm�  �          A��
?���h  AH��B"��C���?����(�A"ffA���C�s3                                    BxX|l            A��H?�=q�k33AHz�B!G�C���?�=q���A!A�C�O\                                    BxX�  �          A��\?���jffAH��B!C���?����33A"=qA��C�T{                                    BxX��  �          A��\?��H�iAH��B"�C��R?��H���HA"�HA�=qC���                                    BxX�^  g          A��R?�
=�iG�AJ=qB#�C��?�
=����A$Q�B Q�C��H                                    BxX�  A          A��\?�G��g�
AJ�HB$�C��R?�G���  A%p�B�C��f                                    BxXŪ  �          A�Q�?���g33AJ�RB$33C�<)?������A%��B��C���                                    BxX�P  �          A��@�\�h��AF�HB!  C��q@�\���A!A�C��3                                    BxX��  �          A�=q@	���ip�AF�HB �C�1�@	����=qA!�A�G�C���                                    BxX�  �          A��@(��h��AEB 
=C�G�@(����
A!�A���C��R                                    BxX B  �          A���@��iADz�B  C��@���{A�
A��HC���                                    BxX�  �          A��?�(��h��AF=qB ��C�` ?�(����A!�A���C��                                    BxX�  �          A��H?�p��c33AJffB%�C�xR?�p��~=qA'
=B��C��                                    BxX,4  �          A�p�?�(��c�AK33B%�HC���?�(��~�RA((�B�
C���                                    BxX:�  �          A�G�?�=q�c33AK\)B&\)C��?�=q�~{A(z�Bz�C�K�                                    BxXI�  �          A�Q�@�jffA>�RB(�C��=@����A33A�33C�)                                    BxXX&  �          A���?��H�b{AHz�B%��C���?��H�|Q�A&=qB
=C��f                                    BxXf�  �          A��?xQ��dQ�AE��B#=qC��{?xQ��}�A#33B�RC��H                                    BxXur  �          A�33?�{�e��AB�RB p�C�1�?�{�~�HA Q�B =qC��H                                    BxX�  T          A��?����e�AC
=B C�� ?����~�HA ��B ��C�^�                                    BxX��  T          A��?�G��hz�A@(�Bp�C�u�?�G����\A�A�33C�"�                                    BxX�d  T          A��?�Q��f=qADz�B!�C�\)?�Q��33A"�RBz�C��                                    BxX�
  
�          A���?���`  AK\)B(z�C��H?���yA*�\B�C�\)                                    BxX��  
Z          A��?���]p�AL��B*C�XR?���w\)A,��BffC��                                    BxX�V  T          A�p�?�Q��]AL��B*ffC���?�Q��w�A,��B=qC��=                                    BxX��  �          A���?�  �ZffAO
=B-p�C��)?�  �tQ�A/�Bz�C�U�                                    BxX�  �          A���?�(��YAO�B.�C���?�(��s�A0z�BQ�C�J=                                    BxX�H  �          A���?n{�\Q�AMp�B+�RC��?n{�u�A.{B
=C��q                                    BxX�  �          A��H?��
�\z�AMG�B+z�C�#�?��
�uA.{B
=C��                                    BxX�  �          A�=q?��H�_\)AG�
B&��C�|)?��H�w�
A(��B�\C�=q                                    BxX%:  �          A�z�?�  �c\)AD��B#�C�H?�  �{33A%�B
=C�Ф                                    BxX3�  
�          A��H?fff�dQ�AD��B"��C��\?fff�|  A%p�B�HC���                                    BxXB�  �          A�G�?s33�c\)AG33B$��C���?s33�{33A(  B�C��q                                    BxXQ,  T          A�{?   �ep�AAG�B ffC�  ?   �|z�A"{B��C���                                    BxX_�  �          A�Q�?G��i�A<(�B�C���?G���{A��A�p�C�ff                                    BxXnx  �          A�z�?�G��g�A?�B�C��q?�G��~{A z�B ��C�Ф                                    BxX}  �          A�G�>����ep�AE�B"��C��>����|z�A&�\B��C���                                    BxX��  �          A��
?��hQ�AC\)B (�C�
=?��
=A$��BQ�C��3                                    BxX�j  "          A��?0���iG�AABC�XR?0����A#33B(�C�:�                                    BxX�  �          A�?W
=�k33A?\)B\)C���?W
=���\A ��A��
C�~�                                    BxX��  �          A���?�  �m�A<(�BffC���?�  ��G�Ap�A�ffC���                                    BxX�\  "          A�?xQ��lz�A=��B��C�޸?xQ�����A33A�33C��
                                    BxX�  �          A�\)?5�l��A<  B�\C�aH?5��
=AA�p�C�C�                                    BxX�  �          A�\)?Q��k
=A>=qBC��R?Q���(�A z�B {C�u�                                    BxX�N  
�          A��?s33�k�A>=qBp�C���?s33��ffA ��A��C��3                                    BxX �  �          A��?z�H�l(�A=��B��C��?z�H���\A (�A�
=C��                                     BxX�  �          A��?�{�mp�A;�
B{C�%?�{��
=A�\A�  C���                                    BxX@  �          A�G�?�z��n{A9B�\C�8R?�z����A��A�\)C��                                    BxX,�  �          A��\?�
=�l��A9��B{C�H�?�
=��Q�A��A���C��                                    BxX;�  �          A�(�?���n�\A5�BffC�xR?������AQ�A��
C�H�                                    BxXJ2  �          A�z�?����j�HA;
=B�C���?����~�RA�RA��RC�j=                                    BxXX�  �          A�\)@ff�s
=A1G�BffC��@ff����AQ�A��HC��=                                    BxXg~  �          A��@�k�
A;33B��C�f@��A\)A�C��R                                    BxXv$  T          A�p�@��pQ�A5G�B
=C��\@�����A�A��HC��f                                    BxX��  S          A���@33�s�A1�B{C��3@33����A��A�G�C��\                                    BxX�p  �          A�  ?����x��A+�B	�\C��3?�����33A
=A��\C�XR                                    BxX�  �          A��R@Q��y��A,z�B	��C���@Q�����A  A��C���                                    BxX��  �          A�  @
=�z�RA(��BC���@
=���
AQ�A��
C��R                                    BxX�b  �          A��?�33�z=qA)��BC�t{?�33���Ap�A�{C�=q                                    BxX�  �          A���?�(��w�A+�
B
�C���?�(���Q�A(�A�33C�b�                                    BxXܮ  �          A���@���|��A#�Bp�C��@����ffA�A�Q�C��{                                    BxX�T  T          A��@
=�{
=A&=qB(�C��3@
=����A
�RA�{C��R                                    BxX��  T          A�\)@�xz�A)p�B33C���@��z�AffA��\C���                                    BxX�  
�          A��@   �xQ�A)G�B=qC���@   ��Q�AffA���C�q�                                    BxXF  �          A���?�\)�w\)A)G�B��C�s3?�\)��A�RA�ffC�=q                                    BxX%�  "          A�G�?���q�A5�B{C��?����33A\)A�G�C���                                    BxX4�  �          A��?�\)�q�A5G�B��C��?�\)���A�A��C�ٚ                                    BxXC8  �          A�z�?����n=qA6ffB�C��?����\)Ap�A�{C��q                                    BxXQ�  �          A�?��
�j{A9BG�C���?��
�{�A!G�BffC�U�                                    BxX`�  �          A��?�\)�k33A<z�BG�C���?�\)�|��A$(�B�\C�|)                                    BxXo*  �          A��?�33�f{AC
=B C�H�?�33�x(�A+\)B
33C��                                    BxX}�  
�          A�p�?�33�hz�A@��Bz�C�E?�33�z=qA)p�B{C�)                                    BxX�v  T          A��
?�  �n=qA;33BG�C�e?�  �33A#\)B�C�=q                                    BxX�  "          A��?����s�A4Q�B��C���?������A(�A�G�C�`                                     BxX��  T          A���?�(��t  A6�\B�RC�� ?�(���(�A�\A��C��3                                    BxX�h  �          A��?\�p(�A5�B{C��?\��{A�\A�
=C��{                                    BxX�  "          A�=q?��H�hQ�A=p�BffC�c�?��H�x��A&�HB  C�:�                                    BxXմ  �          A�z�=�Q��R=qAQ��B3��C�33=�Q��d��A=�BG�C�/\                                    BxX�Z  �          A�Q�?���g�
A>=qB=qC�?���xQ�A((�B(�C��                                    BxX�   �          A�33?�=q�n�RA8(�B(�C��=?�=q�~�\A!��BQ�C�b�                                    BxX�  T          A�
=?���l��A;
=B��C�?���|��A$��BG�C���                                    BxXL  �          A��?�  �g
=AD��B!z�C��q?�  �w�
A/\)B��C���                                    BxX�  �          A�z�?h���k33AA��B��C��f?h���{\)A,  B	Q�C���                                    BxX-�  �          A��
?\(��f�\AE�B!��C��{?\(��w
=A0(�B��C��
                                    BxX<>  �          A��
?���d(�AH(�B$��C��?���t��A3�B��C�f                                    BxXJ�  �          A��\?��dQ�AJ{B&  C��?��t��A5��B33C��R                                    BxXY�  �          A�zᾮ{�X��AV�\B2�
C�E��{�j=qAC33B=qC�S3                                    BxXh0  
Z          A�{��(��V{AXz�B533C���(��g�AE��B!C�%                                    BxXv�  �          A��?�ff�k�
A:�HB=qC��?�ff�z�RA&ffB
=C��=                                    BxX�|  �          A�?�Q��m�A;\)B�C�J=?�Q��|��A'
=B�C�(�                                    BxX�"  
�          A�{?���mG�A=p�B�C��?���|  A)G�BQ�C���                                    BxX��  �          A���?��
�p(�A;\)Bz�C�p�?��
�~�\A'33B�
C�L�                                    BxX�n  �          A�Q�?�
=�o33A;�B�C�B�?�
=�}p�A'�B�C�!H                                    BxX�  �          A�Q�?�\)�n{A=�B�\C�%?�\)�|z�A)��BG�C�f                                    BxXκ  �          A�=q?�\)�n{A<��BG�C�'�?�\)�|Q�A)G�B33C��                                    BxX�`  �          A��H?�z��p(�A<Q�B{C�4{?�z��~=qA(��B33C��                                    BxX�  �          A��\?xQ��iG�AC�B�C��?xQ��w�A0��B��C�˅                                    BxX��  T          A�p�>�\)�a��AIB'33C���>�\)�pz�A8  B��C���                                    BxX	R  "          A��
>u�b�HAIB&�C�y�>u�q��A8  B�C�q�                                    BxX�  �          A���>����e��AJ{B%ffC���>����t(�A8Q�B33C���                                    BxX&�  T          A�G�>�\)�d��AK\)B&ffC��>�\)�s�A9�BffC��                                    BxX5D  �          A��?z��l(�AJ=qB"33C�q?z��zffA8Q�BffC�                                    BxXC�  �          A���?p���tQ�ADQ�B�C�?p������A2{B
Q�C���                                    BxXR�  �          A���?���pQ�AIG�B�C�'�?���~{A7�B=qC�
                                    BxXa6  �          A�{>�33�n�HAH��B (�C���>�33�|z�A7\)B�HC���                                    BxXo�  �          A�{>aG��o\)AHz�B��C�n>aG��|��A733B�RC�g�                                    BxX~�  T          A�p�?:�H�n�HAF�\BC�c�?:�H�|  A5p�B�HC�Q�                                    BxX�(  �          A���?n{�n�\AD��BC��=?n{�{�A3�
B{C��3                                    BxX��  �          A���?��m�AF�HB�C�  ?��z{A6ffBffC��3                                    BxX�t  �          A�ff?�R�n=qAD(�B�
C�1�?�R�z�HA3�
Bz�C�"�                                    BxX�  �          A��\?!G��n{AE�BffC�8R?!G��z�\A4��B=qC�(�                                    BxX��  �          A���?�  �p��AA��B�
C��?�  �}�A1p�B�HC�Ф                                    BxX�f  
�          A��H?����tz�A<(�B�RC��q?�����{A,  B  C���                                    BxX�  �          A�  ?���s�A;
=Bp�C��H?���
=A+
=B�HC��                                     BxX�  �          A�(�?�{�v�RA7
=B�RC���?�{���HA'33BQ�C���                                    BxXX  �          A�(�?��{
=A0z�B��C�XR?�����A z�A���C�5�                                    BxX�  �          A��@ff����A+\)BC���@ff���A33AC���                                    BxX�  �          A�z�@�
���A&=qB ffC���@�
��Q�A{A�33C�p�                                    BxX.J  �          A�z�@ ����p�A   A�{C�p�@ ����  A�
A��HC�S3                                    BxX<�  �          A�ff@33��=qAG�A�p�C�|)@33����A�A֏\C�`                                     BxXK�  �          A�ff?��R���AQ�A�33C�Z�?��R���AQ�AΣ�C�AH                                    BxXZ<  T          A�ff@33��p�A�A�Q�C�u�@33����A	G�A�(�C�Z�                                    BxXh�  �          A��\?��R����A��A�  C�W
?��R��A	G�A�(�C�=q                                    BxXw�  �          A�ff?�����A��A�{C�
?����A	p�AЏ\C�                                      BxX�.  
�          A�=q?����A��A�C�=q?���33A
�\Aҏ\C�&f                                    BxX��  �          A�Q�@�����A  A���C�k�@����A	�A�  C�S3                                    BxX�z  "          A�=q@����AA�C�n@���
=A
=A�G�C�T{                                    BxX�   �          A�  ?�
=���HAA�=qC�C�?�
=���RA\)A�(�C�,�                                    BxX��  �          A�(�@z���p�A  A��HC�z�@z���33A	��A�33C�b�                                    BxX�l  T          A��@���\)A
=A�C���@���
=A��A�z�C�y�                                    BxX�  �          A��@�����A�A�z�C�  @����\A	��AхC��                                    BxX�  �          A��@(����
=A\)A��
C���@(�����RA�A�G�C�e                                    BxX�^  "          A��R@0  ���A�A�{C��q@0  ��33A  A��C��q                                    BxX	
  �          A��R@1G���p�AG�A�ffC���@1G���
=Az�Aޏ\C���                                    BxX	�  �          A�ff@1���z�A�HA��C��
@1���{AffA�z�C���                                    BxX	'P  T          A�=q@=p����HA�
A���C��@=p���ffA\)A�C��q                                    BxX	5�  T          A��@=p���G�A�A��C��@=p�����A33A�33C��=                                    BxX	D�  T          A�(�@G���AG�A���C�J=@G����HA�A�z�C�+�                                    BxX	SB  �          A��@7����Ap�A�C��H@7����RAp�A�p�C��                                    BxX	a�  
�          A��@!G�����A�HA�G�C�AH@!G���A
=AɅC�*=                                    BxX	p�  �          A�(�@3�
���A�A�\)C���@3�
��(�A	��A��C��                                    BxX	4  T          A�z�@0����{A{A���C�� @0����
=A�HA��C��f                                    BxX	��  �          A�z�@#33��ffAz�A�{C�q�@#33��p�A��A�\)C�XR                                    BxX	��  T          A�ff@���(�A=qA��C��q@���33A�A��C��f                                    BxX	�&  @          A�  @5��A
=A�ffC�� @5��ffA (�A�Q�C���                                    BxX	��  �          A�33@  ��=qA
=Aޏ\C�Ф@  ����AQ�A���C���                                    BxX	�r  "          A��
?����G�A{A�33C�=q?�����A�
A�C�.                                    BxX	�  T          A��H?ٙ�����A��A�Q�C���?ٙ���\)A
=A�33C��=                                    BxX	�  �          A�33?˅����A=qA�ffC��f?˅��G�A��Aݙ�C��R                                    BxX	�d  �          A�z�?�Q���Q�A(�A��C���?�Q����HA
�RAۅC��f                                    BxX

  �          A�(�?�{��=qA��A���C��q?�{����A\)A̸RC���                                    BxX
�  �          A��H?���z�A
=A�
=C��{?����HA��A�G�C�Ǯ                                    BxX
 V  �          A���?�Q���=qA�HA�\)C��H?�Q����\AA��C���                                    BxX
.�  �          A�=q?����\)Az�A���C��H?�����A�AӮC���                                    BxX
=�  �          A�?�{����AA�C�XR?�{����A	�A��HC�O\                                    BxX
LH  �          A�p�?�������A��A��C��?������
A(�Aܣ�C��                                    BxX
Z�  T          A��?�������A��A�p�C�{?������
A��A�G�C��                                    BxX
i�  �          A��?�G����AffA�(�C�33?�G���G�AffA�Q�C�(�                                    BxX
x:  �          A�p�?�  ��p�A��A�
=C��)?�  ����A��A݅C���                                    BxX
��  T          A��?У�����A  A�C��3?У����AQ�A܏\C�Ǯ                                    BxX
��  �          A�p�?\��{A{A��C��H?\��{A
�\A��
C��R                                    BxX
�,  T          A��?У����AG�A�(�C�Ǯ?У���\)A�A�C���                                    BxX
��  �          A��?�G�����A  A�=qC��?�G���p�A��A�{C��R                                    BxX
�x  
�          A�33?�����A��A�Q�C�q?�����A{A�z�C�3                                    BxX
�  �          A�ff?�z���(�A  A�{C�L�?�z����AG�Aҏ\C�B�                                    BxX
��  �          A�\)?����A�
A�\)C�(�?�����Ap�A�=qC��                                    BxX
�j  �          A�p�?�����33A  A�\)C�(�?������HA��Aԏ\C�                                      BxX
�  T          A�p�?\��p�A�
A�  C��?\���AA߅C��q                                    BxX
�  �          A���?��H���A��A��RC�"�?��H��G�A�HA�z�C��                                    BxX\  �          A�p�?������A�RA��C�(�?�����33A�A�  C�                                      BxX(  �          A��?�G���(�Ap�A噚C��?�G�����A  A�{C��                                    BxX6�  T          A�G�?�33��=qA�
A�33C�E?�33����A
�\A�  C�<)                                    BxXEN  "          A�33?���{A(�A��
C�.?����A
=A���C�%                                    BxXS�  T          A���?�
=��33A�A�=qC�\)?�
=��z�A(�AݮC�S3                                    BxXb�  �          A��
?�������A��AC�Ff?�����=qA��A�\)C�>�                                    BxXq@  
�          A�
=?��H���
A
=A�{C�q�?��H���A�\A�(�C�h�                                    BxX�  �          A�\)@{����Az�A��C��@{����A(�A�=qC�޸                                    BxX��  �          A���@Q���AffA���C��q@Q�����A{A�C��{                                    BxX�2  �          A�Q�?�
=�{�
A"ffB�
C��R?�
=�~ffA�\A���C���                                    BxX��  �          A�z�?�p��~�\A�\A�z�C���?�p���ffA�HA��
C���                                    BxX�~  |          A���?��H�z�HA$��B��C���?��H�}G�A!p�B��C��H                                    BxX�$  "          A��
?�=q�}p�A{A���C��R?�=q��A�RA��\C���                                    BxX��  �          A�G�?���~�HA�A�Q�C�� ?����ffA�RA�\C�|)                                    BxX�p  "          A���?�\)�yA"ffB��C��{?�\)�{�A�B �HC��                                    BxX�  �          A�?�33�|  A�B �
C��)?�33�}A��A��\C��
                                    BxX�  
�          A��?�G��zffA   B�RC�7
?�G��|  Ap�A��\C�1�                                    BxXb  �          A��R?�Q��uA$z�BG�C���?�Q��w\)A!�B
=C��3                                    BxX!  �          A�z�@33�o�A,Q�B�
C��f@33�q�A*{BC��                                     BxX/�  �          A��H?�{�n�RA/\)BffC���?�{�p(�A-G�BffC��f                                    BxX>T  q          A���?��l  A3
=BC��{?��mp�A1�B��C��\                                    BxXL�  �          A���@��m�A0(�BffC���@��nffA.ffBC���                                    BxX[�  T          A��@���p��A,  B
=C��@���r{A*ffB�C��                                    BxXjF  
�          A�
=@���p��A,  B�C�(�@���q�A*�\B�\C�%                                    BxXx�  �          A��R?�z��t��A'33B�
C��H?�z��uA%B��C��                                     BxX��  
�          A��?�G��l��A,��B��C�c�?�G��m��A+�B�C�`                                     BxX�8            A���?���iG�A/\)B��C��3?���i�A.ffB�HC���                                   BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�*              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�v              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXh              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX(�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX7Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXF               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXT�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXcL              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXq�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�>              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�0              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�|              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�"              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXn              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX!�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX0`              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX?              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXM�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX\R              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXj�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXy�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�D              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�6              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXт              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�(              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�t              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX)f              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX8              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXF�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXUX              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXc�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXr�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�J              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�<              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXʈ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�.              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�z              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX"l              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX1              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX?�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXN^              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX]              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXk�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXzP              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�B              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXÎ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�4              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�  �          A�
=@N{���H@�  A�Q�C��3@N{����@�33A�
=C���                                    BxX�&  Z          A�33?����@�
=AŅC��R?���~�RAG�AָRC��                                     BxX�  �          A�
=?�����\@�\)A���C���?����=q@�33A�=qC�ٚ                                    BxXr  
�          A���?+���z�@�{A��C�,�?+���{A ��AՅC�33                                    BxX*            A��R�u�\)@��RA�(�C����u�z=qA	G�A��C��\                                    BxX8�  h          A����8Q���{@�p�A�ffC����8Q���@�A�ffC��\                                    BxXGd  �          A�{?Tz����R@���A�{C�u�?Tz���z�@�p�A�=qC�|)                                    BxXV
  "          A���?��
��  @�{A�  C�7
?��
��@��HA�Q�C�@                                     BxXd�  T          A��H?��H���@���A��\C���?��H��\)@�ffA�
=C��{                                    BxXsV  |          A��R?�����@ȣ�A��RC�*=?�����@�ffA�\)C�8R                                    BxX��  �          A�=q?�p����@�=qA��RC�l�?�p����@�  A��C�z�                                    BxX��  T          A��
?�
=��ff@�ffA��C�` ?�
=��{@�(�Aģ�C�o\                                    BxX�H  �          A�{?��H����@�ffA�\)C�k�?��H��Q�@�z�Aď\C�z�                                    BxX��  "          A�Q�@   ��(�@�  A�=qC�q�@   ��(�@�{A��C�~�                                    BxX��  "          A���@����@��A�
=C���@�����@�  A��\C��                                    BxX�:  "          A��R?�(���Q�@vffAJ�HC�� ?�(����H@��HAr=qC��                                    BxX��  T          A�{@1G���G�?�p�@ϮC��{@1G���z�@.{A\)C���                                    BxX�  T          A�G�@333���@C33A\)C��H@333���@s33AG33C���                                    BxX�,  �          A��
@.�R��{@>�RA�HC�~�@.�R���H@o\)AC
=C���                                    BxX�  "          A�33@0  ��p�@A�A�C���@0  ��(�@r�\AFffC���                                    BxXx  T          A���@7
=��33@6ffA��C��q@7
=��{@g�A=��C�Ǯ                                    BxX#  �          A�G�@<(���p�@3�
A�\C���@<(���=q@eA;�C��f                                    BxX1�  "          A�G�@Dz����@%�AffC��@Dz���z�@W�A/�
C�)                                    BxX@j  
�          A���@Dz���33@$z�A=qC��@Dz���(�@W
=A/�
C��                                    BxXO  �          A�G�@N{���@=q@��\C�O\@N{��z�@L��A'33C�Y�                                    BxX]�  T          A��H@P����\)@p�@�
=C�` @P����Q�@AG�AC�j=                                    BxXl\  
�          A�ff@[����R@   @љ�C���@[����
@3�
A33C��{                                    BxX{  �          A��@k���  ?�ff@��HC��@k���33@
=@�Q�C�#�                                    BxX��  
�          A�{@N{����?�ff@�\)C�y�@N{��=q@ff@�C��H                                    BxX�N  "          A�33@J�H��\)@@�
=C�w
@J�H��Q�@HQ�A*�HC���                                    BxX��  �          A�Q�@QG���  @*�HA�\C���@QG�����@]p�A>ffC���                                    BxX��  �          A���@Y����{@0��A
=C���@Y�����H@c�
AC
=C��
                                    BxX�@  
�          A�G�@fff��33?��@ϮC�H�@fff��Q�@+�A  C�T{                                    BxX��  �          A��@=p��}G�@QG�A7�C�L�@=p��z=q@���AdQ�C�\)                                    BxX�  "          A��H@p��x��@�{A|  C��H@p��t��@��RA���C��3                                    BxX�2  T          A��@4z��x��@`  AG�C�!H@4z��u�@���At��C�1�                                    BxX��  �          A
=@3�
�r�H@z=qAc�C�<)@3�
�o�@�p�A�z�C�O\                                    BxX~  h          A{�
@!��o�@�33Aq�C��@!��k�
@��A�  C��H                                    BxX$            A�33@1G��y�@{�A_\)C�
=@1G��u��@�
=A��RC�)                                    BxX*�  �          A��
@s33��{?�ff@a�C�|)@s33���?��@���C���                                    BxX9p  
�          A��@|(����?\(�@9��C��R@|(���p�?��H@�G�C���                                    BxXH  �          A��@s�
���?�R@ffC���@s�
��
=?�(�@�Q�C��=                                    BxXV�  �          A�=q@5���>�{?�33C��H@5���p�?���@��C���                                    BxXeb  �          A���?�(���\)��=q�l��C�^�?�(������W
=�333C�]q                                    BxXt  �          A��H@(Q�����?�(�@��C���@(Q�����@A ��C��3                                    BxX��  "          A���@2�\��?�p�@أ�C��\@2�\���R@7
=Az�C��R                                    BxX�T  �          A�z�@.�R��(�@H��A,Q�C���@.�R����@�  A\��C���                                    BxX��  T          A�Q�@)����\)@mp�AL(�C��f@)���33@��\A|��C��
                                    BxX��  T          A�\)@>{�z�R@��A���C�]q@>{�v{@�ffA��C�w
                                    BxX�F  T          A�33@Vff�zff@�=qA
=C�
=@Vff�v{@�p�A��C�%                                    BxX��  T          A��@X���|��@�  A^ffC�\@X���x��@�33A��C�'�                                    BxXڒ  �          A���@;���ff@HQ�A-C�*=@;��}@�Q�A_\)C�9�                                    BxX�8  �          A���@G��~�\@U�A9�C���@G��{33@��RAk�C���                                    BxX��  �          A��\@L(��}�@o\)AP��C���@L(��yp�@��A�G�C��f                                    BxX�  T          A���@XQ��{�@�  A_�C�\@XQ��w�@��
A��RC�'�                                    BxX*  "          A�
=@b�\�|��@q�AR{C�P�@b�\�y�@���A�  C�h�                                    BxX#�  �          A��@g��}G�@fffAG�C�q�@g��y��@�\)Ay�C���                                    BxX2v  �          A��@`���}�@�Q�A]C�=q@`���y@���A�{C�U�                                    BxXA  
�          A���@j=q�|��@���Ax��C���@j=q�xz�@��A�C��                                     BxXO�  
�          A�
=@Tz��w
=@�Q�A�ffC��@Tz��qp�@�(�A��C�5�                                    BxX^h  T          A�33@/\)�l��@��RA��C�>�@/\)�eG�A��A���C�j=                                    BxXm  
�          A���@J=q�Yp�A{B�C���@J=q�Pz�A"=qBz�C���                                    BxX{�  "          A��@Dz��A�A2=qB'33C�#�@Dz��6�\A=�B4  C��=                                    BxX�Z  
�          A��
@0���E�A4��B&�RC�G�@0���;33A?�
B3��C��H                                    BxX�   T          A�{@5�=�A8Q�B-�HC�� @5�2{AC
=B:��C�&f                                    BxX��  T          A���@B�\�A�A/33B%\)C�{@B�\�6�\A:{B2G�C�xR                                    BxX�L  
�          A��\@:�H�%p�AI�BFffC��H@:�H�G�AS33BSG�C�xR                                    BxX��  T          A�G�@AG��ȣ�Am�B��3C��)@AG���z�Ar�\B���C��H                                    BxXӘ  
�          A�(�@C33�ۅAk\)B{�HC���@C33��\)Aqp�B�
=C��                                    BxX�>  T          A�z�@33��AS
=BT{C��3@33��RA[�
BaQ�C�u�                                    BxX��  "          A�p�?�=q�0��A@z�B<�\C�{?�=q�%�AJ�RBJ  C�]q                                    BxX��  �          A���@,(��-AA�B<��C��3@,(��!�AK
=BI�
C�o\                                    BxX0  T          A�G�@6ff�-G�A@  B;�HC�aH@6ff�!p�AI�BI{C���                                    BxX�  T          A�p�@>{�(  ADQ�BA\)C���@>{�  AN{BN�\C�xR                                    BxX+|  �          A�(�@3�
�$  AJffBG�C���@3�
��AT  BU(�C�@                                     BxX:"  
�          A��H@A��$(�AK33BG�\C�9�@A���AT��BTC��)                                    BxXH�  �          A��H@:�H��RAO�
BM�HC�5�@:�H��AX��B[�C���                                    BxXWn  �          A�G�@,�����AX��BY��C�q@,���33Aap�Bg=qC��                                     BxXf  �          A���@>{���AX��B[�C�{@>{��Aap�Bh�RC��\                                    BxXt�  �          A��H@E�*ffAK
=BCp�C�3@E�AU�BP��C���                                    BxX�`  �          A���@C33�8Q�A>�\B3��C�j=@C33�,Q�AIp�BAz�C��                                    BxX�  T          A��@P���>{A4��B*{C��\@P���2�\A@  B7��C�&f                                    BxX��  
�          A���@Q��B=qA0  B$C���@Q��7
=A;�B2Q�C��                                    BxX�R  !          A��@W��9p�A8  B.�C��@W��-AC33B<
=C���                                    BxX��  "          A�p�@U�K�A$(�BC�Y�@U�@��A0Q�B%p�C���                                    BxX̞  �          A�\)@U�Lz�A"�RBQ�C�T{@U�B{A/
=B$
=C��{                                    BxX�D  �          A��@l���H��A%�B�\C�8R@l���>{A2{B'33C���                                    BxX��  �          A�z�@g
=�G�
A*ffB=qC��@g
=�<��A6ffB*�C��H                                    BxX��  "          A�  @g
=�K�
A$z�BQ�C��@g
=�A�A0��B%
=C�P�                                    BxX6  
�          A�@c33�O33A33B�C��=@c33�D��A+�
B�C��                                    BxX�  �          A�p�@Tz��Qp�A��B�HC�  @Tz��G33A)��BC�y�                                    BxX$�  T          A�\)@p���S
=A�B
�\C��{@p���I�A$z�BffC�S3                                    BxX3(  �          A�G�@��H�QA�HB	�C���@��H�G�
A#�
B��C�3                                    BxXA�  T          A�\)@��R�Q�A�B�
C���@��R�H(�A"�HB�\C�L�                                    BxXPt  T          A�33@y���R�HA=qB	\)C�@ @y���H��A#33B33C���                                    BxX_  �          A���@��R�U��A33BQ�C��H@��R�L(�AQ�B�C�"�                                    BxXm�  �          A�
=@�=q�W33A
ffA��RC�aH@�=q�M�A�B{C���                                    BxX|f  �          A��H@���Z�\A�\A�C�� @���Q��A(�B��C��R                                    BxX�  T          A��H@c�
�Z{Az�A���C�T{@c�
�P��A{B�
C��                                    BxX��  �          A��R@b�\�QG�Az�B�C���@b�\�G33A%��B��C��                                    BxX�X  �          A�z�@G
=�HQ�A$(�B
=C��R@G
=�=p�A0��B(33C�Y�                                    BxX��  �          A�Q�@<���<  A4(�B+�\C�@<���0(�A?�
B9�RC���                                    BxXŤ  T          A�Q�@@���c\)@��A�\C��)@@���Z�HA��B ��C�5�                                    BxX�J  �          A�(�@3�
�j�\@�=qAʣ�C�j=@3�
�b�H@��A�p�C��R                                    BxX��  �          A��\@-p��jff@�  AυC�=q@-p��bffA�RA�ffC�k�                                    BxX�  �          A���@J�H�f�R@�\A�p�C�1�@J�H�^ffA�
A�(�C�k�                                    BxX <  �          A��H@Tz��aA��A陚C���@Tz��X��A�
B(�C��H                                    BxX�  T          A���@U��f�\@��HA�G�C�� @U��^=qA  A�  C���                                    BxX�  �          A���@Vff�k33@��
A�  C�j=@Vff�c�@���A���C���                                    BxX,.  �          A��@
=�P(�A�RB�
C�%@
=�E�A#�
BffC�g�                                    BxX:�  T          A��
@E�\(�@�
=A�C�T{@E�S�Ap�BQ�C���                                    BxXIz  T          A�  @S33�ap�@��
A�C��{@S33�Yp�A(�A���C���                                    BxXX   �          A�33@Q��g\)@߮A��
C�e@Q��_�@��A�RC��)                                    BxXf�  �          A�33@6ff�p��@��HA��C�\)@6ff�j=q@��A�=qC��H                                    BxXul  �          A��R@'
=�qG�@�
=A��
C���@'
=�j�H@�A��C�
=                                    BxX�  �          A���@.�R�q@�33A�ffC�q@.�R�k�@�=qA��C�@                                     BxX��  T          A�G�@4z��r�R@��\A�
=C�E@4z��lQ�@љ�A�Q�C�g�                                    BxX�^  �          A��@&ff�s\)@���A�\)C��R@&ff�m�@ϮA��RC��R                                    BxX�  "          A���@!��u��@��HA��HC���@!��o�
@�=qA�=qC���                                    BxX��  T          A�=q@7��n�R@�=qA��C�s3@7��h(�@���A���C���                                    BxX�P  �          A�=q@N{�p��@��A�ffC��@N{�j�H@���A��C�1�                                    BxX��  �          A�@QG��r�R@��RA�
=C��@QG��mG�@�A�(�C�8R                                    BxX�  T          A��\@P���mp�@��A�{C�0�@P���f�H@��
A�G�C�]q                                    BxX�B  |          A���@Z�H�p(�@�
=A�
=C�h�@Z�H�jff@�A�(�C��\                                    BxX�  �          A�p�@c33�b�H@�p�AʸRC�f@c33�[33@��\A�C�B�                                    BxX�  
9          A��@O\)�p��@EA5C�3@O\)�mG�@�=qAp(�C�+�                                    BxX%4  "          A{�@Vff�t�Ϳ�{��G�C�*=@Vff�up���G��ǮC�&f                                    BxX3�  �          A�p�@^{�t���6ff�$��C�aH@^{�w\)�������C�P�                                    BxXB�  
�          A�{?�ff�&ff�P���L�C�:�?�ff�3��Ep��=�RC��                                    BxXQ&  �          A��?�p��%p��P���MG�C�3?�p��2�\�Ep��>z�C�Ǯ                                    BxX_�  �          A��?���'�
�N�H�J�
C��f?���4���C��<  C��H                                    BxXnr  �          A�
=?\�-��Hz��C��C��q?\�9��<���4�
C��R                                    BxX}  "          A��?��R�&�\�I���I  C�b�?��R�3\)�>=q�:�C�&f                                    BxX��  �          A��?8Q��=q�Up��]��C�@ ?8Q���
�K\)�O{C�\                                    BxX�d  T          A�p�?   �ff�UG��]��C��{?   �   �K\)�O
=C�q�                                    BxX�
  �          A�ff?Tz��=q�I��N��C�ff?Tz��+
=�>ff�?�HC�8R                                    BxX��  T          A~=q?Y���p��J=q�R��C���?Y���&=q�?��C�C�W
                                    BxX�V  �          A��H?�{�=q�U��`C��3?�{��
�L(��Q�C�C�                                    BxX��  �          A�  ?�����R�M��VQ�C��q?����#�
�C��G�C�T{                                    BxX�  �          A��?����#��F�R�Iz�C�&f?����0(��;��:��C��                                    BxX�H  T          A�?����9��9���3Q�C��
?����E���-��$p�C��                                    BxX �  "          A�G�?�p��0���E���@�C�.?�p��=��9�1G�C��R                                    BxX�  �          A���?k��/\)�HQ��B�
C�b�?k��<  �<z��4  C�9�                                    BxX:  T          A��?n{�6{�Bff�;�C�XR?n{�Bff�6=q�,G�C�1�                                    BxX,�  
�          A��H>���>{�9G��1
=C���>���I���,���"33C���                                    BxX;�  �          A�z�>�{�=G��9G��1�C��{>�{�H���,���"��C�Ǯ                                    BxXJ,  �          A�=q>����;��:ff�3G�C��>����G33�-�$p�C��H                                    BxXX�  �          A�(�>W
=�;��:{�2��C��H>W
=�G\)�-p��$�C�y�                                    BxXgx  �          A�(�>�
=�=G��8Q��0�HC�>�
=�H���+��"{C���                                    BxXv  "          A�ff=��7
=�?33�8��C�L�=��C
=�2�H�*(�C�G�                                    BxX��  "          A�G�?&ff�9G��9���4  C���?&ff�D���-G��%=qC��H                                    BxX�j  �          A�33>aG��4Q��>ff�:=qC���>aG��@(��2ff�+p�C���                                    BxX�  �          A��R=��
�6ff�4���2�
C�33=��
�A�(z��${C�0�                                    BxX��  �          A���?5�=���.{�*{C���?5�HQ��!���\)C���                                    BxX�\  T          A�\)?�R�J�H���C�ff?�R�T���ff�

=C�U�                                    BxX�  �          A~{=�Q��Qp��  �
{C�33=�Q��Z=q�=q���RC�0�                                    BxXܨ  �          Aw\)��ff�j=q��ff���\C��R��ff�nff�`  �R�\C�                                    BxX�N  �          A{�?����2�R�/
=�0�C��{?����=p��#33�"�C�~�                                    BxX��  �          A�Q�?c�
�C\)�&=q�!G�C�{?c�
�Mp��p���C���                                    BxX�  T          A�=q?fff�F=q�!���C��?fff�P(�����{C��R                                    BxX@  �          A�?xQ��G\)�
=���C�9�?xQ��Q��{�ffC��                                    BxX%�  T          A�?�
=�G�
�=q�{C���?�
=�Qp��G��
�\C���                                    BxX4�  T          A~�\?J=q�G������C��?J=q�Q��z��
ffC���                                    BxXC2  T          A}��?����D����R�
=C�}q?����N�\����C�`                                     BxXQ�  �          A}��?�
=�;33�(���'  C��?�
=�Ep�������C��                                     BxX`~  �          A~�R@Q��B�R�\)�{C��f@Q��LQ���R��
C�C�                                    BxXo$  �          A~�\?�z��C33� �����C�޸?�z��L���Q����C��\                                    BxX}�  �          A~�H?����B�R�!�����C�B�?����Lz����G�C�\                                    BxX�p  T          A~{?�R�D��� Q���\C�t{?�R�Nff���33C�b�                                    BxX�  �          A~�H>\�E� ���Q�C��H>\�O��  ���C��
                                    BxX��  "          A~ff>8Q��G33��\�  C�l�>8Q��P������C�g�                                    BxX�b  T          Ay��\�P������C�⏿�\�X�����R���HC���                                    BxX�  
�          Av�R�8Q��Z�\�����\)C��H�8Q��aG��ə�����C��                                    BxXծ  
�          Av=q>�G��S����
��p�C���>�G��Z�R��������C��                                    BxX�T  "          Ax(�?��H�]G���z���G�C��H?��H�c��������HC�o\                                    BxX��  
�          Ar�R?�
=�_����H��
=C�k�?�
=�d����
=���RC�]q                                    BxX�  T          Ao
=?�p��f{�r�\�k�C�s3?�p��iG��9���333C�j=                                    BxXF  	�          Ap��?�33�l���z��C���?�33�n�\��33���HC���                                    BxX�  
C          Ap��?��l������{C�C�?��n�\������
C�>�                                    BxX-�  1          Aqp�?��\�j�R�L���D  C�w
?��\�mp���\�  C�p�                                    BxX<8   �          Ar{@���b�R����z�C���@���f�H�s�
�i��C�s3                                    BxXJ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXY�   �          Aq�?�\�h���dz��[\)C�y�?�\�k��+��$  C�n                                    BxXh*              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXv�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�v              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�h              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXδ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX	L              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX&�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX5>              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXC�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXR�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXa0              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXo�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX~|              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�"              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�n              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXǺ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�`              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXR              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX.D              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX<�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXK�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXZ6              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXh�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXw�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�(              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�t              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX��  `          Adz�?���]�Dz��G�C�"�?���`  �
=���C��                                    BxX�f  �          Aa�?!G��_�
��\)��33C�L�?!G��`�Ϳh���l(�C�J=                                    BxX�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX�X            A_�?����\Q���R�33C���?����]��ff��33C���                                    BxX	�  T          A`Q�?�G��\���	����C��)?�G��^{���H��\)C��R                                    BxX�  �          A`��?��R�]�����Q�C���?��R�^�\�\��ffC���                                    BxX'J  �          A`��?��R�]p�� ����
C��\?��R�^�H��=q��{C���                                    BxX5�  �          A`(�?����\(��
=q�C��?����]����p��\C�                                      BxXD�  "          A`��?���[\)����G�C���?���\�ÿ�p���\C��3                                    BxXS<  �          A`(�?�p��Z=q�"�\�&�RC�"�?�p��[�
�����ffC��                                    BxXa�  T          A`��?��H�\���
�H�{C��?��H�^=q��G����C��q                                    BxXp�  �          Aa�?���^ff��
�{C�ٚ?���_���33��p�C���                                    BxX.  T          Ac\)?�33�^{� ���"�RC�c�?�33�_�
������C�]q                                    BxX��  
�          Ab�R?�=q�\(��8Q��;�C��=?�=q�^{��R���C��H                                    BxX�z  
�          Ac\)?��^ff�(��ffC�o\?��`  ��ff��Q�C�h�                                    BxX�   �          Abff?��]�� ���#�C�q�?��^�R��\)��33C�k�                                    BxX��  
�          Af�R?��H�Y���(����C���?��H�]���  ���RC��                                     BxX�l  
�          Af�H?��\�Yp��������\C��?��\�\���������
C���                                    BxX�  �          AhQ�?���Z�R��{���HC���?���]���=q��=qC���                                    BxX�  
�          Ahz�?��Y��������
C��R?��\����G���p�C���                                    BxX�^  �          AhQ�?�33�Y����z����C��?�33�\��������G�C��f                                    BxX  �          Ah��?�=q�[����H����C�K�?�=q�^�\�~�R�}C�@                                     BxX�  �          Ai?�\)�\����p�����C��q?�\)�_��s�
�q�C�Ф                                    BxX P  �          AiG�?У��[��������RC�c�?У��^�\��������C�W
                                    BxX.�  �          Ah��?���Z�H������33C���?���^{������C���                                    BxX=�  T          Ah(�?����Zff��ff��33C�
=?����]p����
��C���                                    BxXLB  	�          Ah  ?����Z{��p���(�C�W
?����]����\����C�K�                                    BxXZ�  
(          Ag�
?�=q�Y����\)����C�Q�?�=q�\z�������C�Ff                                    BxXi�  	�          Aj=q?�
=�Y���R���HC��q?�
=�\����z����C���                                    BxXx4  "          Aip�?��Z=q��  ��(�C��
?��]G�����\)C���                                    BxX��  �          Ai�?�(��Z�H�����=qC��?�(��]��  ���C�f                                    BxX��  T          Aip�?�p��\����Q���  C��?�p��_��|���{33C�f                                    BxX�&  �          Ah��?�p��\  ��������C�3?�p��^�R�~{�}�C�
=                                    BxX��  T          Aip�?��H�Yp��������HC��
?��H�\z���
=����C��=                                    BxX�r  �          Af�R@z��M��(��ɮC��3@z��Qp����
���C��                                     BxX�  "          Ag�@\)�;�����{C��q@\)�@����(����RC��R                                    BxX޾  4          Af�\@��A��\)���\C��q@��F=q��  ��33C�                                    BxX�d  �          Ag33@=q�F�H��\)��
=C�}q@=q�J�H��  ���C�b�                                    BxX�
  �          Ag\)@
=�Bff��{��{C�~�@
=�F�R��\)���C�aH                                    BxX 
�  �          Af�\@�\�AG���\)����C�]q@�\�E���������
C�@                                     BxX V  �          Aip�?���:ff�
{�{C�?���?\)�
=��RC��=                                    BxX '�  T          Ai�?�33�;\)�
=q��C�ff?�33�@Q��\)���C�O\                                    BxX 6�  �          Aip�?�\)�:�H�
�H�  C���?�\)�?��  �	C���                                    BxX EH  �          Ah��?z�H�<���
=��C�]q?z�H�Ap�� (��C�N                                    BxX S�  �          Ah(�?��
�>�H�ff�C�?��
�C\)��\)� �RC��)                                    BxX b�  �          Ag�?�p��A��=q��HC���?�p��E���z����
C��R                                    BxX q:  �          Ag�?�p��;\)�{�p�C��
?�p��@  ��
=���C��H                                    BxX �  �          Af�H?��H�@z����
�G�C�� ?��H�D����R��
=C�Ф                                    BxX ��  �          Ag
=?����@�������HC��=?����E���ff��z�C�}q                                    BxX �,  �          Ag33?�
=�>{����Q�C��)?�
=�B=q���H����C���                                    BxX ��  �          Af�H?����C
=��\����C�b�?����F�H�����C�Q�                                    BxX �x  �          Ae�?fff�A�����R��RC�#�?fff�Ep�������RC�R                                    BxX �  T          Ae��?�z��Ap���(�� �\C���?�z��EG�����\C���                                    BxX ��  �          Ad��?L���B�H��{����C��H?L���F�\�ᙚ��ffC�ٚ                                    BxX �j  T          Ad��?W
=�@����z��G�C��)?W
=�Dz������Q�C��3                                    BxX �  �          Adz�?@  �?���  �z�C���?@  �C33��(�����C���                                    BxX!�  �          AdQ�?J=q�<(�� ���	33C��=?J=q�@  ���G�C��H                                    BxX!\  �          AdQ�?Tz��8z���p�C��?Tz��<z�� (����C�H                                    BxX!!  �          Ad��?�=q�4  �
=��HC�\)?�=q�8  ����=qC�J=                                    BxX!/�  �          AdQ�?�=q�3��
�H��
C�H?�=q�7\)����G�C��                                    BxX!>N  �          Adz�?����.=q�  �z�C�3?����2=q�
=�{C���                                    BxX!L�  �          AdQ�?����,Q���R�   C�޸?����0Q����C�                                    BxX![�  �          Ad��@��'\)���&Q�C��H@��+��
=� (�C���                                    BxX!j@  �          AdQ�@ ���,  �ff��C�H�@ ���0  ����C�+�                                    BxX!x�  �          Ad  ?�ff�,z��=q��\C���?�ff�0Q������\C��                                     BxX!��  �          Ad��?���0����R�z�C�:�?���4Q��	���\C�%                                    BxX!�2  �          Ad��?Ǯ�1���R�G�C��?Ǯ�4���
{�p�C��\                                    BxX!��  �          Ad��?��H�/\)�(��ffC�l�?��H�2�H����RC�U�                                    BxX!�~  �          Ad��?��
�.�R����(�C��H?��
�2ff�z���\C��=                                    BxX!�$  �          AdQ�?��0(��{��C��3?��3��	�z�C���                                    BxX!��  �          Ad  ?��0������p�C���?��4  �z��{C���                                    BxX!�p  �          Ac�
?��1�33��\C���?��4���
=�Q�C��3                                    BxX!�  �          Ac�
?��H�0z������\C�
=?��H�3�����ffC��3                                    BxX!��  �          Ac\)?�33�.�R�=q��HC��\?�33�1��
=q��
C�ٚ                                    BxX"b  �          Ac\)?����0���33�  C�  ?����4  �33�
=C��                                    BxX"  �          Ac\)@ ���-�
=�
=C�@ @ ���0���\)�=qC�(�                                    BxX"(�  �          Ab�H@
=�*�R������C���@
=�-�����C��H                                    BxX"7T  �          Aa��?���/
=�33�G�C��?���1����C��\                                    BxX"E�  �          Aa�?�=q�.�R�
�R�=qC���?�=q�1���\)�C���                                    BxX"T�  �          A`��?����/
=�
=q��C��
?����1����R�G�C��                                    BxX"cF  T          A`��?�ff�.�R�
�R�ffC���?�ff�1G�����C��)                                    BxX"q�  �          A`z�?޸R�/��	p���HC��H?޸R�1��{��RC�q�                                    BxX"��  �          A`��?����,���Q����C��=?����/��	����C��R                                    BxX"�8  �          A`��?޸R�,(������C��R?޸R�.�\�
=��RC���                                    BxX"��  �          A`��?�\)�,z��G����C���?�\)�.�H�
ff���C��R                                    BxX"��            A`��?��R�,(�����Q�C�9�?��R�.ff�
{���C�(�                                    BxX"�*  �          A`Q�?���,z��z��33C���?���.�R�	���C��f                                    BxX"��  �          A_�?�  �*�R���z�C��=?�  �,���\)�
=C��)                                    BxX"�v  �          A`  @   �+��(����C�J=@   �-�	�����C�:�                                    BxX"�  �          A_�
@Q��*ff���G�C��=@Q��,Q��
�\�{C���                                    BxX"��  �          A_�@�\�)����
=C�R@�\�+��
=q��C�f                                    BxX#h  �          A_\)@\)�(���{��C�  @\)�*�\��
���C��\                                    BxX#  �          A_33@=q�(��������C�n@=q�*�\�
�H���C�]q                                    BxX#!�  �          A_\)@���((�����ffC���@���)�����C��H                                    BxX#0Z  �          A^�R?�Q��/�
�{���C�]q?�Q��1p��  �Q�C�T{                                    BxX#?   �          A_
=?���1�����Q�C�8R?���2�\��H���C�0�                                    BxX#M�  �          A^�R?��R�0z����C���?��R�1��  �Q�C���                                    BxX#\L  �          A^=q?���+��
=q���C�� ?���,�������C���                                    BxX#j�  �          A^{?��,  �	���\C��)?��-p��\)�ffC��3                                    BxX#y�  �          A]��?�=q�.�R�{��HC�  ?�=q�/�
�z����C�R                                    BxX#�>  �          A^�\?����.{�\)���C���?����/33���
=C���                                    BxX#��  �          A]�@ ���+����Q�C�K�@ ���,���=q��C�B�                                    BxX#��  �          A\��@�+\)���=qC��@�,Q��=q��\C�}q                                    BxX#�0  �          A[�?�  �0����\)�Q�C���?�  �1�����C��
                                    BxX#��  T          AZ�H?��H�1G����
���C�?��H�2{�����33C���                                    BxX#�|  T          A[�@(��,����\�(�C���@(��-p������
C��3                                    BxX#�"  
�          A[�
@���+����33C�n@���,Q���R�
=C�g�                                    BxX#��  �          A[33@���.ff� Q��\)C��@���/
=��
=�G�C���                                    BxX#�n  �          AY��?���2�\��z��C��\?���3
=���H���C���                                    BxX$  �          AY�?
=q�5���ff�=qC�^�?
=q�5������p�C�]q                                    BxX$�  �          AX��?��1��\)�33C�]q?��2=q��ff�
�C�]q                                    BxX$)`  �          AX��?E��/������HC�H?E��/�
��(��G�C�H                                    BxX$8  �          AW�?����0z����	�C��)?����0�����H�	{C���                                    BxX$F�  �          AXQ�?��/
=��Q��(�C�XR?��/33�����
C�W
                                    BxX$UR  �          AY�?�p��*�R�33��C��)?�p��*�H�
=��C���                                    BxX$c�  �          AW�?ٙ��2�R���ffC�U�?ٙ��2�R���\)C�U�                                    BxX$r�  �          AX  @�\�,����=q��C�W
@�\�,�����\�C�XR                                    BxX$�D  �          AX  @z��'��  ��RC��3@z��'��(����C��3                                    BxX$��  �          AW�
@��"�H�����RC��@��"�\�	��
=C�
=                                    BxX$��  T          AW�@�\�"=q�����HC�Z�@�\�!��	G��Q�C�]q                                    BxX$�6  �          AW�@   ���33� ��C�@   �33���!�\C��                                    BxX$��  �          AX(�@4z���H�(��-  C�U�@4z��ff����-�RC�\)                                    BxX$ʂ  �          AV{@ff�$Q��  �=qC�p�@ff�#�
����{C�u�                                    BxX$�(  �          AUG�@{�$���ff���C��@{�$z��33���C�
                                    BxX$��  �          AVff@%��Q�����5��C��q@%����G��6�HC��                                    BxX$�t  �          AP��@p��z��G��0�\C���@p����{�1��C��                                    BxX%  �          AN�R@Q��p��
=�533C���@Q��z���
�6�\C��                                    BxX%�  ;          AO�
@���
�H��\�9��C��@���	��\)�;{C�'�                                    BxX%"f  �          AS
=@�R�����9=qC�@�R������:�
C�)                                    BxX%1  �          AQ�@ ���	p���
�?\)C��{@ ���(�����A�C���                                    BxX%?�  �          AH  @{������R���C��f@{�  ��G��z�C���                                    BxX%NX  �          AH��@�\�����G���C���@�\��
���
��C�˅                                    BxX%\�  �          AG�@ff�z���ff��RC��@ff�\)����HC�)                                    BxX%k�  �          AF�H@�\�33��33�Q�C�y�@�\����ff���C��f                                    BxX%zJ  �          AHQ�@��������ffC��R@��Q������
C��                                    BxX%��  �          AJ�H?���'
=�����=qC�R?���%�����
C�"�                                    BxX%��  �          AK\)?���!���p���C�B�?��� z���G���
C�N                                    BxX%�<  �          AK\)@   �=q��
=���C���@   ������H�p�C���                                    BxX%��  �          AJ�\?�\�\)�����C��?�\�������C��                                    BxX%È  �          AH��?У������ 
=C�� ?У���33�#33C��                                    BxX%�.  �          AH��?ٙ���
����&
=C�7
?ٙ����33�)G�C�H�                                    BxX%��  �          AI��@�R�����z��(�C���@�R�
=� z���\C��                                    BxX%�z  �          AI��@��
=��\)���C�Y�@�����z���C�n                                    BxX%�   �          AH��@��\)���R�\)C���@��G�����
=C��
                                    BxX&�  �          AI@Q��z���\)���C�O\@Q��=q�=q� C�ff                                    BxX&l  �          AH  ?��H�z���R�)\)C�"�?��H�{�	G��-Q�C�<)                                    BxX&*  �          AH��?�{����33�/��C���?�{�
=��3�RC��                                    BxX&8�  �          AHz�?�{�Q�����,�C�3?�{����0\)C�*=                                    BxX&G^  �          AG\)?�\)�\)�  �,{C��?�\)����
�R�0z�C�7
                                    BxX&V  �          AD��?�\)�"�R��  �p�C��R?�\)� ���ָR�  C��                                    BxX&d�  �          AC�?�ff�-���G��׮C�n?�ff�+33�������C�xR                                    BxX&sP  �          AF=q?���$Q�����G�C���?���!�����
�C���                                    BxX&��  �          AD  ?�{�0(���
=����C�޸?�{�.ff���R�ˮC��                                    BxX&��  �          AA�?xQ��3���p���G�C�w
?xQ��2{��p�����C�}q                                    BxX&�B  �          AB�H@'��'�
���
���
C�f@'��%�������(�C�)                                    BxX&��  �          AD  @J�H�*�H������(�C�B�@J�H�(�������̣�C�Y�                                    BxX&��  �          AF{@x���&�\��{�иRC�@ @x���$z���ff��G�C�`                                     BxX&�4  T          AF�R@��R�%p������Ώ\C��@��R�#33�����33C�33                                    BxX&��  �          AF�\@y���(����
=��33C�!H@y���&�R�����Q�C�AH                                    BxX&�  �          AF�\@���%������ʣ�C�  @���#����\���
C�C�                                    BxX&�&  �          AEG�@�\)�*ff������C��{@�\)�(Q�������
=C��3                                    BxX'�  �          AE�@���33��\)��p�C���@�������  ���HC��                                    BxX'r  �          AG\)@��
����\)�\)C�L�@��
�Q����(�C��                                    BxX'#  �          AF=q@�\)�ff��{�  C�ff@�\)�
=��ff�
=C���                                    BxX'1�  �          AC33?��"�\�����=qC�Z�?����ָR���C�u�                                    BxX'@d  �          AC�
?�  �0(���z���
=C�:�?�  �-����
=���HC�G�                                    BxX'O
  �          AC�
?Q��5���������C��?Q��3���(����RC�3                                    BxX']�  �          ADQ�@z��&�\�������C���@z��#\)��
=����C���                                    BxX'lV  �          AC33?�p��3
=���
��  C�?�p��0z���
=���RC��\                                    BxX'z�  �          AB�\?�  �3
=��=q����C�/\?�  �0z�������C�9�                                    BxX'��  T          AD��?��H�(z�������C�G�?��H�%G���������C�b�                                    BxX'�H  �          AC\)@
=�-����H�ŅC�#�@
=�*=q���R���HC�=q                                    BxX'��  �          AB�R@�/�
��=q���C��@�-G���ff����C��                                    BxX'��  �          AB{@��.{��z���Q�C���@��+33������Q�C���                                    BxX'�:  �          AA�@
=�,  ��z����C���@
=�(��������\)C���                                    BxX'��  �          AB�H@��)�����R��p�C�q�@��&=q��33���C���                                    BxX'�  �          AB�R?����6�H��  ��C���?����4z������HC��{                                    BxX'�,  �          AC�?#�
�5G���G���z�C��H?#�
�2ff��
=���C���                                    BxX'��  �          AC33?L���7
=��p����C�  ?L���4Q������p�C��                                    BxX(x  �          AA�?s33�5��(���\)C�e?s33�3
=���\��\)C�o\                                    BxX(  �          AB�R?s33�4z����R��  C�ff?s33�1p������Q�C�p�                                    BxX(*�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX(9j   J          AD��@4z��/
=���
���RC�=q@4z��+���=q���C�`                                     BxX(H  �          ADQ�@��3\)������Q�C�� @��0(����
��33C���                                   BxX(V�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX(e\              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX(t              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX(��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX(�N              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX(��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX(��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX(�@              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX(��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX(ڌ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX(�2              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX(��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX)~              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX)$              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX)#�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX)2p              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX)A              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX)O�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX)^b              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX)m              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX){�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX)�T              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX)��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX)��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX)�F              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX)��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX)Ӓ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX)�8              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX)��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX)��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX**              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX*�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX*+v              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX*:              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX*H�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX*Wh              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX*f              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX*t�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX*�Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX*�               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX*��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX*�L              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX*��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX*̘              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX*�>              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX*��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX*��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX+0              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX+�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX+$|              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX+3"              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX+A�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX+Pn              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX+_              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX+m�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX+|`              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX+�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX+��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX+�R              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX+��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX+Ş              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX+�D              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX+��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX+�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX, 6              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX,�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX,�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX,,(              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX,:�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX,It              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX,X              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX,f�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX,uf              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX,�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX,��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX,�X              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX,��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX,��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX,�J              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX,��   �          A:�\@e��(���[����RC�]q@e��"�R��ff��\)C���                                    BxX,�  �          A;
=@R�\�-��.�R�X  C�k�@R�\�(���r�\���C��=                                    BxX,�<  �          A;
=@h���)���S�
���C�|)@h���#\)��33���RC�Ф                                    BxX-�  �          A:{>���8�׿Y����\)C�f>���6ff� ����
C�
=                                    BxX-�  �          A9p�@J�H�(Q��dz����C�aH@J�H�!�������Q�C��{                                    BxX-%.  �          A9�@G��+
=�H���{�
C�"�@G��$�����R����C�j=                                    BxX-3�  �          A9��@0  �+\)�_\)��Q�C�4{@0  �$�����\���C�|)                                    BxX-Bz              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX-Q               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX-_�   {          A9��@!G��2ff��p��G�C�aH@!G��.{�H���{33C���                                    BxX-nl  �          A8��@���1p����"�RC�{@���,���L(����RC�:�                                    BxX-}  �          A8��@/\)�*=q�c33��33C�9�@/\)�#33�����z�C��                                    BxX-��  �          A9�@/\)�*ff�a���Q�C�33@/\)�#\)�������C�~�                                    BxX-�^  �          A9�@)���&ff�������RC�%@)���{�����ޏ\C��H                                    BxX-�  �          A8z�@G
=�%�w���\)C�\)@G
=�{��
=��
=C��q                                    BxX-��  �          A8��@�\)� ���`  ��G�C�H@�\)������\����C�}q                                    BxX-�P  �          A8Q�@C�
�(���\(���\)C�R@C�
�!�����\��C�n                                    BxX-��  �          A8  @Mp��(  �Z�H��=qC��H@Mp�� ���������C���                                    BxX-�  �          A8  @P  �'��]p���  C��H@P  � Q���33����C��q                                    BxX-�B  �          A:ff@�(��"�H�c�
��Q�C��@�(�������p�C�"�                                    BxX. �            A9��@�G��$���\(����C��3@�G�����\��  C�%                                    BxX.�  �          A9G�@w
=�$���c�
��\)C�E@w
=�p����R��{C��R                                    BxX.4  �          A9�@l(��&{�]p�����C��=@l(���R��(�����C�5�                                    BxX.,�  T          A9G�@`���&�\�c33��\)C�Q�@`����H��\)��G�C��q                                    BxX.;�  �          A9@�{�%���G
=�y�C�f@�{��R��G���C�t{                                    BxX.J&  �          A:=q@��� ���3�
�`Q�C�y�@���ff�}p�����C���                                    BxX.X�  �          A:{@�G��$(��@  �o�C��\@�G��p�������C�b�                                    BxX.gr  �          A8z�@�.ff�6ff�eC�\@�'�
���
��C�J=                                    BxX.v  �          A8��@   �.�R�0  �\��C�p�@   �(Q��������C��                                    BxX.��  �          A8z�?У��3���\�"�RC�q?У��.�\�W
=���C�=q                                    BxX.�d  �          A8��?�(��2�\�p��0z�C�?�(��,���a����\C�,�                                    BxX.�
  �          A8z�?�  �2�R�
�H�-C�u�?�  �-G��`  ��C���                                    BxX.��  �          A8��?�{�4z���\�#
=C�o\?�{�/33�Y������C���                                    BxX.�V  �          A9G�?��4Q���
�#�C�8R?��.�H�Z�H��p�C�XR                                    BxX.��  �          A9?˅�6=q�Ǯ��\)C��R?˅�1�<���k�C��                                    BxX.ܢ  �          A:�\?�p��7\)��z���ffC���?�p��333�4z��_�
C�                                    BxX.�H  �          A:�R?�33�8�ͿE��qG�C�u�?�33�5�����-C���                                    BxX.��  �          A:�H?�Q��9p���Q��G�C���?�Q��7�
�����=qC��3                                    BxX/�  �          A;
=?����9p��u��z�C�c�?����7���Q��p�C�n                                    BxX/:  |          A:{?��R�7�
>��@33C�
?��R�733�����C��                                    BxX/%�  �          A;�@��\��{�{C�+�@�33� ���-Q�C��
                                    BxX/4�  �          A;�
@@  �{��=q�
=C�XR@@  �����ff�/C�h�                                    BxX/C,  �          A<��@C33����������C�ٚ@C33�
�H���H�ffC���                                    BxX/Q�  �          A;�@B�\� Q����H��(�C�q�@B�\��
���H�p�C�)                                    BxX/`x  �          A8  @H���-��Q��+\)C��@H���'33�aG����C�^�                                    BxX/o  �          A8  @!��%p��tz���  C��H@!��  ��z���Q�C�Ff                                    BxX/}�  |          A<z�@���������C�33@�������(��5�HC�R                                    BxX/�j  �          A=G�@I���G���{��Q�C�
@I���
�H������HC��
                                    BxX/�  �          A=��@~{��������33C���@~{�G���p����C���                                    BxX/��  �          A=��@�33�!�����
��\)C���@�33��H����C�c�                                    BxX/�\  �          A;�
@=p��\)��33���
C�C�@=p��ff��z��
p�C��{                                    BxX/�  |          A;�?����*�\����=qC��\?�����H�\���C��3                                    BxX/ը  �          A<(�?E��,Q����H��{C��?E�� ����Q���ffC�0�                                    BxX/�N  �          A;�
?�G��'
=��=q��G�C�)?�G��ff��ff��RC�p�                                    BxX/��  �          A9�?��'����\����C���?���
��\)��G�C��q                                    BxX0�  �          A9�?����,z��p  ��G�C��H?����"�\��ff���
C�,�                                    BxX0@  �          A8z�@��+\)�n�R��
=C�^�@��!p����ծC���                                    BxX0�  �          A8��@
�H�+\)�p  ��\)C���@
�H�!G���ff��(�C��                                    BxX0-�  �          A9�@ ���,���dz���  C�AH@ ���#33������\)C��\                                    BxX0<2  �          A8��?�(��+33�w�����C�8R?�(�� �����H��z�C��                                    BxX0J�  �          A8Q�?�33�)p������\)C�3?�33�ff��=q��\)C�k�                                    BxX0Y~  �          A8(�@�
�*�R�p  ��(�C�xR@�
� z���\)��ffC�Ф                                    BxX0h$  T          A8  @\)�-G��;��l  C�y�@\)�$����{��{C�˅                                    BxX0v�  �          A8��@���,���aG���  C�� @���"�R��G��θRC�
                                    BxX0�p  �          A;
=?�
=�,(���  ��Q�C�l�?�
=� ����Q����C��q                                    BxX0�  �          A;
=?��R�-G����
���C���?��R�!������G�C�1�                                    BxX0��  �          A<Q�@!G��(����=q����C��q@!G��(������{C�AH                                    BxX0�b  �          A<��@�H�$Q������ՙ�C�� @�H�{��  ��\C�9�                                    BxX0�  �          A;�@:=q�\)��G����C�Y�@:=q�  ����G�C�4{                                    BxX0ή  �          A:�\@Tz��Q��Ǯ�ffC���@Tz��  ��G��!=qC��
                                    BxX0�T  �          A9p�@���p����(�C���@���z���  �(Q�C���                                    BxX0��  �          A:{?�(��  �������C�H?�(�����{��\C��3                                    BxX0��  �          A<(�?ٙ�� ����=q���C���?ٙ��G������
C�O\                                   BxX1	F  �          A:�R?�G��+���
=��z�C��\?�G���R��G����C��                                    BxX1�  �          A;�?�G��$  �����p�C�0�?�G������\)�(�C��q                                    BxX1&�  �          A;�?�  �33�׮���C��H?�  �������/33C��                                    BxX158  �          A;
=@�R�����{�{C��@�R���  �4�C��)                                    BxX1C�  �          A8(�?fff�  �Ӆ���C��?fff�{��ff�0C�&f                                    BxX1R�  �          A2�H�*=q��
��(�����C~���*=q��\��=q���C}W
                                    BxX1a*  �          A3�
�33�!p������ffC�AH�33�Q���  ��C���                                    BxX1o�  �          A4  � ����R���
��\)C�E� �������33�	�RC���                                    BxX1~v  �          A3��Q��G���=q��\C���Q��
{��  ��C~�\                                    BxX1�  �          A3��!����ff����CB��!��
�R��z��Q�C}�                                     BxX1��  �          A4�׿�p���
��=q���HC��R��p��33��  � �C�9�                                    BxX1�h  h          A2{�<����������C}(��<����\��z�� �\C{�3                                    BxX1�  �          A3
=�33��������33C�=q�33�	G���Q���C                                    BxX1Ǵ  �          A4Q쿠  ������H��  C�B���  ����G��!p�C���                                    BxX1�Z  �          A4�׿8Q��(������
Q�C��׿8Q������/(�C�s3                                    BxX1�   
�          A4(����z���\)�C�ff���ff�����+�C���                                    BxX1�  �          A3
=�
=�G����(�C�q��
=���R��=q�*\)C�                                    BxX2L  �          A4(���\�p���p��\)C��H��\��z�� Q��5�C�                                    BxX2�  �          A4Q쿮{��H��
=���C��=��{���R�p��7�RC���                                    BxX2�  �          A4�ÿc�
�
�R��{�  C�ÿc�
��(��(��C�C��{                                    BxX2.>  �          A4Q쿨�����
����7�RC�{������ff�=q�\��C���                                    BxX2<�  �          A2ff�HQ���{��(��-�Cw33�HQ��\�  �P{Cr�                                    BxX2K�  �          A1��޸R����33���
C�� �޸R��p���Q��#\)C��=                                    BxX2Z0              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX2h�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX2w|              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX2�"   z          A8��@���{�{�F{C���@���G���=q����C�l�                                    BxX2��  �          A7�@��H��\�����(�C�5�@��H�	p����H��(�C�U�                                    BxX2�n  
�          A7\)@�\)����XQ����C�Ф@�\)�	p���p��ˮC�ٚ                                    BxX2�  �          A8(�@s�
����Q����C��R@s�
���љ��C�q                                    BxX2��  �          A6�H@#33�z���ff��Q�C�� @#33��\��  �(�C�q�                                    BxX2�`  �          A8Q�@^{��R��(���=qC�  @^{�������
=C�XR                                    BxX2�  �          A5G�@C33� (��������C�y�@C33��\���R��
=C�5�                                    BxX2�  �          A2=q?�33�"�\��ff���C��R?�33�z������ 33C�f                                    BxX2�R  |          A2ff?Y���'�
�c�
��p�C�N?Y���\)���\��p�C�}q                                    BxX3	�  �          A3�?�z���R��{�ՙ�C��q?�z��
�\�љ��p�C�k�                                    BxX3�  r          A/33@n{�=q�8Q��s�C�P�@n{����=q��33C���                                    BxX3'D  |          A1�@������=q�K�C�@���z��{���ffC��                                    BxX35�  �          A.{@�Q��
=�.�R�ip�C�J=@�Q��	������C�(�                                    BxX3D�  �          A.ff@�(���H� ���+33C�.@�(��
�H�g
=��C��                                    BxX3S6  �          A/�@5��!��E���33C��
@5����=q��{C�ff                                    BxX3a�  �          A/
=@Z=q�G��B�\��(�C���@Z=q�{��  ��C�>�                                    BxX3p�  �          A.ff@� ���Tz���33C�޸@�����=q��(�C�U�                                    BxX3(  �          A+\)@��#\)�z��H��C���@������
����C�33                                    BxX3��  �          A*�\?p���'\)���
���C��\?p��� (��Y�����C���                                    BxX3�t  �          A,  @|������!��\(�C�@ @|����R��
=��
=C���                                    BxX3�  �          A-G�@=p�� (��,���h  C�<)@=p������\)��Q�C�Ǯ                                    BxX3��  �          A-p�@Q��$��� ���V�RC���@Q���H�����C�0�                                    BxX3�f  |          A.{@
=�"{�B�\��(�C��\@
=�ff��33��ffC��                                    BxX3�  �          A4Q�@�R����p�����
C�O\@�R�׮���H�ָRC��{                                    BxX3�  �          A2�H@�ff��(��n�R��{C��{@�ff������
�ڏ\C�H�                                    BxX3�X  �          A1�@أ���ff�u����C�4{@أ���33�������C�Ф                                    BxX4�  �          A0��@�{��R�0���g�C���@�{�  ��(���ffC�޸                                    BxX4�  �          A0(�@��
���L�����C��@��
�����z���33C���                                    BxX4 J  �          A,��@l(�����  ��G�C��=@l(������������C��)                                    BxX4.�  �          A+
=@QG��=q�P����=qC�Z�@QG����������{C�                                      BxX4=�  �          A*=q@p  �����R�C�C��@p  �(���  ��z�C�P�                                    BxX4L<  �          A+\)@~{����#33�\z�C�:�@~{�
=��=q���\C���                                    BxX4Z�  �          A+�@n�R���$z��]C��H@n�R������
��Q�C�33                                    BxX4i�  �          A,��@w
=�=q�i����=qC�+�@w
=�Q���(���
=C�.                                    BxX4x.  �          A,��@z=q�33���H��{C��H@z=q�  �����Q�C���                                    BxX4��  �          A+
=@�������(���z�C�Q�@����������	{C��\                                    BxX4�z  �          A&�R@@����\)����G�C�Y�@@����p�����5\)C�'�                                    BxX4�   �          A(��@c�
����\)��  C��)@c�
���\����	  C�7
                                    BxX4��  �          A*=q@s�
�  ��
=����C��=@s�
����S33��p�C�%                                    BxX4�l  �          A*{@L������=q�	p�C��f@L�����_\)��  C�W
                                    BxX4�  �          A*�\@C33�{��K�
C��
@C33����
=���C�'�                                    BxX4޸  
�          A+
=@w
=��R���Mp�C��q@w
=�Q���
=��=qC���                                    BxX4�^  �          A,��@�\)�=q�  �@��C���@�\)�(���33��
=C��\                                    BxX4�  �          A,��@�������33�D��C��3@����ff��(���  C�}q                                    BxX5
�  �          A,��@����  ��p��(z�C�S3@�����R�s�
��G�C�\                                    BxX5P  �          A.{@����G���33� ��C�N@����(��p�����
C��                                    BxX5'�  �          A,Q�@�33��H��z��#\)C���@�33��p  ����C�W
                                    BxX56�  �          A,  @�  ������
���C��\@�  ����j=q��G�C�8R                                    BxX5EB  �          A,(�@z=q�����R�*=qC�ٚ@z=q���z=q���\C�z�                                    BxX5S�  T          A,  @U� z�Ǯ�ffC�33@U�(��b�\���HC��=                                    BxX5b�  �          A-�@,(��&ff?�Q�@��HC�<)@,(��&�H�n{���C�8R                                    BxX5q4            A*�R@8���#�
?!G�@Y��C��H@8���"ff������=qC��                                    BxX5�  �          A((�?��
�Q�����M��C�w
?��
���{���C���                                    BxX5��  �          A(��?�  ��H��G����C��?�  ����
�{C��                                    BxX5�&  �          A(  ?�\����u����C�5�?�\�	����R���C��=                                    BxX5��  �          A((��\)�p��`  ��z�C�b��\)��\��{���RC�8R                                    BxX5�r  �          A(z�?E��=q�\����33C�8R?E�����������C�q�                                    BxX5�  �          A(z�>�  ��
�O\)���C��R>�  ���
=��\C��=                                    BxX5׾  �          A)G���ff�"=q�<(���\)C��)��ff�����
=��z�C���                                    BxX5�d  �          A(��>aG��%�L����z�C��q>aG����:�H���\C���                                    BxX5�
  �          A)�@�#���z���=qC�t{@�(��P�����
C��                                     BxX6�  �          A+�
@e�
=q��z�����C�Ff@e���ҏ\�G�C��H                                    BxX6V  T          A,��@]p������p���  C�w
@]p����ƸR���C�˅                                    BxX6 �  �          A/33@mp��p������G�C�
@mp�����θR���C���                                    BxX6/�  �          A-p�@�R��H�r�\����C��
@�R��H��(��G�C���                                    BxX6>H  �          A-G�@,���G������p�C�޸@,����
�\�	G�C���                                    BxX6L�  T          A,��@B�\����}p���Q�C��@B�\�(���z��\)C�Ф                                    BxX6[�  �          A-G�@E�  �����Q�C�>�@E� ����Q��33C��H                                    BxX6j:  �          A-��@'���H��(��ʣ�C�� @'���
�����Q�C��\                                    BxX6x�  �          A.{@����������G�C���@��������=qC��f                                    BxX6��  T          A.ff@(Q���
��(���\)C�� @(Q�����љ���C�Ф                                    BxX6�,  �          A-�@2�\����p���C��f@2�\��33��R�&�RC�
                                    BxX6��  �          A/\)@p���H�љ���C�XR@p��ڏ\�33�CG�C��
                                    BxX6�x  �          A.�\��
=��=q�  �u�C�lͿ�
=�:=q�%Q�Cw�{                                    BxX6�  	�          A-���p�����33�\C�� ��p��p��'�Q�C��                                     BxX6��  �          A,�׾�Q����
�\)��C�˅��Q��=q�'���C��=                                    BxX6�j  �          A-G�?
=��������o\)C�)?
=�O\)�$��z�C�"�                                    BxX6�  �          A+33?��
�����
�R�Z  C��)?��
�����z�Q�C�ٚ                                    BxX6��  �          A&�\?��\��Q��	���_
=C��?��\�p  �ff�C��                                     BxX7\  �          A+33@
=��Q����
�8��C�c�@
=��p����hG�C��                                    BxX7  �          A-p�@4z��������
��HC��R@4z��\��\�MC�q�                                    BxX7(�  �          A-��@H����\�y������C��{@H�����\��Q��	\)C��                                    BxX77N  �          A+�@��\�������\C�<)@��\�=q�L(���ffC��q                                    BxX7E�  �          A*�H@U�� zῸQ����C�.@U��\)�g
=���\C���                                    BxX7T�  �          A+\)@;���H�{�V�RC�9�@;��=q�����33C��f                                    BxX7c@  �          A-�@����������  C��{@��{��p����C�Ǯ                                    BxX7q�  �          A-@<(����XQ���\)C�N@<(��������C�.                                    BxX7��  
�          A.�R@C�
� z��AG���z�C�~�@C�
�������RC�L�                                    BxX7�2  "          A-��@���� �Ϳ�R�Q�C���@�����\�6ff�t  C�S3                                    BxX7��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX7�~   y          A,��@@����I����C�~�@@���ff�������C�Y�                                    BxX7�$  �          A-G�@5�=q�N{���HC�  @5��\�����\C��3                                    BxX7��  �          A-p�@8����
��Q����C�w
@8���z���G����C��)                                    BxX7�p  
�          A.=q@
=���Q���{C��f@
=�p���=q���C��                                    BxX7�  
(          A-�?�(�� (��o\)����C��?�(��ff����ffC�z�                                    BxX7��  T          A/33@�
�z���p����HC���@�
�Q������C��\                                    BxX8b  
�          A/�
?����
=��������C�,�?����33�θR�
=C��                                    BxX8  �          A0Q�?�ff�(�����ѮC���?�ff�=q�޸R��C�<)                                    BxX8!�  T          A0(�?c�
�ff��
=��=qC�� ?c�
�33��G��%�
C��                                    BxX80T  "          A3\)?�33�Q����H��  C�\)?�33�z���{�&�C��{                                    BxX8>�  |          A4��>\��\�������C��>\�����
�1  C�W
                                    BxX8M�  �          A6ff@   �����  ��C��@   �����9
=C�Z�                                    BxX8\F  
�          A8(�?�\)�������HC�L�?�\)��\)���H��C�`                                     BxX8j�  
(          A7�
?c�
�ff��p���C��R?c�
�߮����P(�C��q                                    BxX8y�  �          A8(�?   ��H��ff�\)C���?   ����G��P��C�
=                                    BxX8�8  �          A7�������  ���C��q����{��R��C���                                    BxX8��  T          A9G�?���	p������%��C�o\?����=q�
=�[p�C�z�                                    BxX8��  �          A8��>�  �����H�$33C��{>�  ��ff��R�Z=qC�{                                    BxX8�*  

          A<(�@���������R��z�C��R@�����\)�أ��Q�C�t{                                    BxX8��  
�          A>�R@���R?xQ�@�C�� @��=q���H��33C���                                    BxX8�v  �          A@��A (��	�@��A%p�C���A (��G�=u>��RC��                                    BxX8�  �          AA�@��=q���R���C�0�@��G��^�R��Q�C�f                                    BxX8��  "          A?�@׮�G������
=C�8R@׮����XQ���33C���                                    BxX8�h  �          A@��@�z��&ff������
C�@�z�����p����C���                                    BxX9  "          AA��@�
=�%��@  �ep�C���@�
=����
����C�'�                                    BxX9�  T          A=G�@���"ff�\(���G�C�{@�������  ��
=C�p�                                    BxX9)Z  @          A7�
@z�H�������  C���@z�H�G����H���C��R                                    BxX98   
�          A2�R@,(���
=���\�2\)C��@,(���z��{�eQ�C�@                                     BxX9F�  �          A6=q@P  �,(��p������C�ff@P  �#��_\)��=qC�Ф                                    BxX9UL  �          A5p�@G��-�>���@33C�H@G��*ff�p��4z�C�*=                                    BxX9c�  T          A7�@Vff�.=q�z�H����C��=@Vff�%p��c�
��G�C��R                                    BxX9r�  T          A8  @G��0Q�p����C��=@G��'��c�
��z�C�N                                    BxX9�>  �          A8(�@j�H�-����R��
=C�\)@j�H�#33�tz���33C��                                    BxX9��  "          A6�R@mp��+\)��\)���
C���@mp�� ���z=q��ffC�q                                    BxX9��  �          A6�\@z=q�*�R�����
=C�@z=q�!���e���RC���                                    BxX9�0  �          A8  @tz��,(�������Q�C�Ǯ@tz��"ff�q����C�S3                                    BxX9��  �          A7�@\���,�Ϳ�ff�Q�C��q@\��� �����
��
=C�}q                                    BxX9�|  �          A8Q�@Tz��,���33�8(�C���@Tz��ff�����p�C�Ff                                    BxX9�"  �          A7�@Fff�,���{�2�HC���@Fff�
=��G���C���                                    BxX9��  �          A6=q@C33�-p����H�	�C���@C33�!p�������C�g�                                    BxX9�n  T          A5@A��.=q������=qC���@A��$���o\)���C�7
                                    BxX:            A5@L(��-����=q��{C�1�@L(��$  �mp���ffC���                                    BxX:�  	�          A6=q@N{�-녿�ff���C�B�@N{�$z��l(���p�C���                                    BxX:"`  
E          A6{@Z=q�,�ÿ�������C�� @Z=q�#\)�l����C�9�                                    BxX:1  
�          A6�H@]p��-G��������C���@]p��#��mp����C�U�                                    BxX:?�  
�          A7�
@^{�.=q����=qC�ٚ@^{�$(��tz���p�C�Y�                                    BxX:NR  
�          A7�@S33�/��   �!G�C�Z�@S33�((��Mp����C��{                                    BxX:\�  �          A7�
@O\)�0(��\)�2�\C�33@O\)�(Q��QG���ffC��                                    BxX:k�  �          A8��@G��1�B�\�r�\C���@G��)��`  ��
=C�7
                                    BxX:zD  "          A:�\@Q��2�R�8Q��c33C�+�@Q��*=q�^�R����C��\                                    BxX:��  T          A8  @U��/�
�+��Tz�C�p�@U��'��X����
=C��{                                    BxX:��  "          A:�H@Mp��3\)�L���z=qC��)@Mp��*ff�dz���z�C�`                                     BxX:�6  
�          A<  @Fff�4�׿}p����C���@Fff�*�H�r�\���\C�
                                    BxX:��  T          A<Q�@E�4�ÿ�ff���C��f@E�+
=�vff����C��                                    BxX:Â  
�          A<z�@Z�H�3\)����ffC�z�@Z�H�(���|�����RC���                                    BxX:�(  
�          A<Q�@N�R�4  ������(�C�H@N�R�)G���  ���HC�z�                                    BxX:��  
�          A;�@L(��3\)���\��ffC��3@L(��(z�������C�p�                                    BxX:�t  
�          A;
=@W
=�1��������C�j=@W
=�&�\�������C��3                                    BxX:�  
�          A:�R@333�4  �����z�C���@333�)��vff���RC�e                                    BxX;�  �          A9�@�H�5���z��7�C��@�H�-G��\(����C�J=                                    BxX;f  "          A9�@:�H�333���
���HC�P�@:�H�)��u��ffC���                                    BxX;*  "          A:=q@2�\�4(��p�����C���@2�\�*ff�q���p�C�W
                                    BxX;8�  �          A<z�@:=q�6{�p����=qC�.@:=q�,(��s33���RC���                                    BxX;GX  T          A<��@@  �5���=q��
=C�j=@@  �)��p����
C���                                    BxX;U�  "          A:=q@E�2�\��(�����C��)@E�'�������{C�5�                                    BxX;d�  
Z          A8Q�@E��0(���33��  C��3@E��$z���p����C�XR                                    BxX;sJ  �          A8z�@Z=q�.=q�Ǯ����C��{@Z=q�"{������p�C�P�                                    BxX;��  "          A8  @l���,�Ϳ�33���C�w
@l���!���(���Q�C�
                                    BxX;��  "          A7�
@�Q��*�\�\���HC�P�@�Q���\���R��C�f                                    BxX;�<  �          A;\)@���,  �������
C�S3@���!G��z�H����C���                                    BxX;��  T          A;33@�
=�)��\)��
=C��)@�
=�ff�����z�C��q                                    BxX;��  T          A:�R@�33�)�Ǯ���C���@�33����������C���                                    BxX;�.  �          A8(�@����(�׿�  ��C�\@����������ffC��H                                    BxX;��  
�          A7�@����'�
�����Q�C�R@����ff��  ����C��{                                    BxX;�z  �          A7
=@mp��*{������C�� @mp��(���z����C�j=                                    BxX;�   "          A8Q�@�
=�&�R���"ffC���@�
=�����p���33C��                                    BxX<�  �          A5@}p��'�
���ffC�W
@}p���\��\)����C�%                                    BxX<l  �          A5��@|���'������=qC�P�@|���=q�������C�"�                                    BxX<#              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX<1�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX<@^              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX<O              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX<]�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX<lP              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX<z�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX<��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX<�B              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX<��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX<��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX<�4              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX<��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX<�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX<�&              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX<��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX=r              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX=              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX=*�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX=9d              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX=H
              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX=V�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX=eV              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX=s�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX=��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX=�H              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX=��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX=��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX=�:              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX=��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX=چ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX=�,              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX=��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX>x              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX>              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX>#�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX>2j              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX>A              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX>O�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX>^\              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX>m              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX>{�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX>�N              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX>��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX>��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX>�@              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX>��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX>ӌ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX>�2              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX>��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX>�~              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX?$              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX?�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX?+p              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX?:              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX?H�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX?Wb              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX?f              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX?t�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX?�T              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX?��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX?��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX?�F              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX?��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX?̒              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX?�8              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX?��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX?��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX@*              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX@�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX@$v              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX@3              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX@A�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX@Ph              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX@_              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX@m�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX@|Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX@�               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX@��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX@�L              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX@��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX@Ř              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX@�>              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX@��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxX@�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXA 0              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxXA�  �          A
ff������ff@��B*��C��
������@w�Aԣ�C�ٚ                                    BxXA|  G          A
=��������@���B&��C�AH��������@n{A��C�{                                    BxXA,"  
�          A�ÿ�z����@�33BU�\Cz�q��z����@��RB\)C�\                                    BxXA:�  
�          A�ÿУ���G�@ᙚB^p�Cz0��У���33@�
=BffC��                                    BxXAIn  
�          A�׿����@���BW�HC}� ����=q@�  B  C��H                                    BxXAX  �          Az�z�H���@���BM33C���z�H���@�z�B(�C���                                    BxXAf�  �          A�
��{��\)@���BT(�C��q��{��ff@��HBffC�P�                                    BxXAu`  "          A�H���R��z�@�  BT��C���R��33@��HBz�C��                                    BxXA�  �          A33������@׮BSffC}ٚ���Ӆ@�=qB\)C��                                    BxXA��  �          A(���=q���\@�=qBU33C{����=q��=q@�p�B��C�>�                                    BxXA�R  "          A���Q�����@�33BCp�C��\��Q���(�@���BffC���                                    BxXA��  �          A�
��\)��@��B?�C��
��\)��\)@�A��
C��                                    BxXA��  "          A�þ���Q�@�G�BG�\C�W
����z�@�ffBz�C�}q                                    BxXA�D  �          A���=q��
=@�ffBF��C��쾊=q��\@�(�B��C��                                    BxXA��  �          A녿J=q���@��HBDC�� �J=q�߮@���B�C���                                    BxXA�  
�          A
=�L�����@�\)BIG�C��
�L����{@�ffB�C��{                                    BxXA�6  
�          A33��ff���@��B/�C�׿�ff���@z=qAܣ�C��{                                    BxXB�  �          A�R��=q�Å@�B)��C����=q��\)@l��A��HC��                                    BxXB�  �          A�H��\)����@�{B!�C��`\)��=q@[�A�Q�C��                                     BxXB%(  �          A�H�n{���H@��B+�C��Ϳn{��
=@qG�Aԏ\C�w
                                    BxXB3�  "          A33�@  ���@�\)B*��C����@  ��G�@o\)A�(�C�(�                                    BxXBBt  �          A\)�+�����@���B,�C��f�+���G�@q�A�Q�C�xR                                    BxXBQ  
�          A�
���H��z�@��
B�C��3���H��=q@@��A���C��R                                    BxXB_�  �          A�׿�����@�=qB33C��f����z�@L��A�(�C�                                      BxXBnf  �          A	G�����\)@�G�BQ�C�������R@I��A�Q�C�%                                    BxXB}  �          A�þ\���@��HB�RC�Z�\���@N{A���C��)                                    BxXB��  "          AQ������(�@��\B
=C�C׾������
@N{A���C��=                                    BxXB�X  "          A�׾u��\)@��
A��C�R�u���?�AD  C�/\                                    BxXB��  "          AQ콏\)���
@��Bz�C��{��\)��H@   A�(�C��q                                    BxXB��  �          AQ�?
=��\)@5A���C�*=?
=��?�\@c33C��                                    BxXB�J  �          AQ�>���@H��A�33C�|)>��
=?Tz�@���C�s3                                    BxXB��  �          AG�?s33��
=@z�HA�
=C��q?s33����?�\AJ�\C�u�                                    BxXB�  �          A?�����@�{B
=C���?�����p�@ffA���C�\                                    BxXB�<  �          AQ�?�\)����@��B�C�޸?�\)���@5A�
=C�*=                                    BxXC �  T          A   ?�(���G�@�33B�C��
?�(���@J=qA�ffC�Ǯ                                    BxXC�  �          @��?�Q���33@�=qBH�
C�AH?�Q���(�@�Bz�C�XR                                    BxXC.  �          A (�?�G����\@���B'{C���?�G���33@Y��A���C��                                    BxXC,�  �          A Q�?p����@���B\)C�O\?p����@C�
A�
=C��                                     BxXC;z  
�          @�33?E����@��RB&�\C��?E����@U�A�33C�)                                    BxXCJ   T          @�G�?\(�����@���B*��C�T{?\(���@\��A�=qC��=                                    BxXCX�  �          @�?G���@�Q�B�C��)?G�����@2�\A���C��                                    BxXCgl  T          A�?.{����@QG�A��C�� ?.{� z�?��\@�{C�l�                                    BxXCv  T          A?
=���@<(�A�(�C�@ ?
=� ��?(��@��RC��                                    BxXC��  �          A��?�\)��\@P  A��C�XR?�\)���R?��\@�  C��                                    BxXC�^  T          A Q�?�����@5�A��C��?�����?
=@�  C��{                                    BxXC�  �          A z�?Q���  @/\)A��C�  ?Q���
=>�@Z�HC��                                    BxXC��  �          @�ff<���\)@�Q�BOG�C�4{<���  @��B�
C�'�                                    BxXC�P  T          @�=q?�����Q�@�{B�HC��)?�����Q�@{A�z�C�1�                                    BxXC��  T          @�
=?��H����?ٙ�AD��C�L�?��H��ff���s33C�%                                    BxXCܜ  T          @�(�?�ff��Q�@{A�{C��
?�ff��p�>�z�@ffC��q                                    BxXC�B  �          @�  ?���׮@e�A�  C�}q?����\)?�  A2�\C���                                    BxXC��  �          @���?��\��G�@��
A�33C�q�?��\��ff@�Aw�C���                                    BxXD�  �          @��>�������@��\B�C���>�����p�@>{A��C���                                    BxXD4  �          @��H���H��\)@��BK�C�
=���H�ƸR@�Q�B	�C��q                                    BxXD%�  
�          @�33�\�mp�@��Bg��Cw��\��(�@��\B)
=C~8R                                    BxXD4�  T          @�\)��
=��  @�G�B?\)C�����
=���
@xQ�A�=qC�q                                    BxXDC&  �          @�  �u���\@�Q�B=�C�� �u��{@u�A���C��=                                    BxXDQ�  �          @�
=�!G���\)@�33BEz�C�j=�!G���{@�B�C�9�                                    BxXD`r  "          @�
=�!G�����@�\)BL{C�G��!G����@��B
�C�*=                                    BxXDo  
�          @�G���33�U�@˅Bk�RCpW
��33��Q�@���B/ffCyB�                                    BxXD}�  �          @�����\)�z�@�p�BH��CLǮ��\)�j�H@��\B%�C[W
                                    BxXD�d  
�          @�z������Q�@�
=BE�\CJO\����dz�@�p�B$Q�CX��                                    BxXD�
  
�          @���h���5@�(�BXffCY��h����G�@��B)\)CgB�                                    BxXD��  T          @����P���J=q@�ffB[ffC`\�P�����
@���B({Cl(�                                    BxXD�V  �          @���?\)�u�@�\)BP�Ch{�?\)��ff@��
Bp�CqG�                                    BxXD��  �          @�p��:�H�j�H@˅BVp�Cg���:�H���\@���B�CqO\                                    BxXDբ  �          @�{�<���9��@׮Bi�RC`���<�����R@�(�B5��Cm�H                                    BxXD�H  "          @�ff�ٙ���(�@�G�B^��Cw���ٙ���=q@��HB �C}                                    BxXD��  A          @�
=�	����@�p�BW
=Cr���	�����H@��RB33CyǮ                                    BxXE�  
�          @���G����@��BTG�CxE��G�����@�G�B�\C}�q                                    BxXE:            @�ff������Q�@��HBH=qCy
=�����ə�@�\)B	ffC}ٚ                                    BxXE�  T          @�G����K�@θRBk{CjW
����z�@�G�B1��Ct�R                                    BxXE-�  
�          @����E�  @�  Bq�HCX!H�E���\@��\BB�
Ch�f                                    BxXE<,  
�          @��Y����z�@׮BwCF�{�Y���A�@�(�BU��C]�3                                    BxXEJ�  
�          @���E��z�@ָRBvp�CS���E�o\)@�z�BJG�Cfp�                                    BxXEYx  �          @�
=�)���0  @�ffBq33Cb��)����G�@�z�B<  Co��                                    BxXEh  "          @�33������\)@�(�BN\)C�J=�����ȣ�@���B��C��                                    BxXEv�  T          @��;�ff����@��
B5�\C�����ff����@tz�A�p�C�!H                                    BxXE�j  T          @�33� ���j�H@�\)B^�Ck�
� �����@�p�B$�\Ct޸                                    BxXE�  �          @����333�N�R@�(�Bd�Ce\�333��\)@�{B.\)Cp�H                                    BxXE��  �          A ���hQ쿳33@�
=BsffCI\�hQ��Tz�@���BO�HC^p�                                    BxXE�\  �          @�p��q녾��@��Btp�C;��q��=q@ϮB]p�CT�                                    BxXE�  �          A Q���
�33@�=qB��=C`�f��
���R@�(�BO�RCqG�                                    BxXEΨ  
�          A �׿.{����@ָRBd  C����.{����@�
=B"�RC�Ǯ                                    BxXE�N  �          A (�>���@أ�Bi
=C��=>���@��B'=qC��                                    BxXE��  �          @�>.{���R@�Bg
=C�(�>.{��@�
=B%G�C��3                                    BxXE��  �          @��R�Y���(�@�
=B�ǮCz녿Y�����R@�
=B[��C��H                                    